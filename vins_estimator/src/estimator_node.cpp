#include <stdio.h>
#include <queue>
#include <map>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

#include "estimator.h"
#include "parameters.h"
#include "utility/visualization.h"


Estimator estimator;

std::condition_variable con;//当 std::condition_variable 对象的某个 wait 函数被调用的时候，它使用 std::unique_lock(通过 std::mutex) 来锁住当前线程。当前线程会一直被阻塞，直到另外一个线程在相同的 std::condition_variable 对象上调用了 notification 函数来唤醒当前线程
double current_time = -1;
queue<sensor_msgs::ImuConstPtr> imu_buf;//队列是一种线性表，先进先出，入队imu_buf.push(),出队:imu_buf.pop()
queue<sensor_msgs::PointCloudConstPtr> feature_buf;
queue<sensor_msgs::PointCloudConstPtr> relo_buf;
int sum_of_wait = 0;

std::mutex m_buf;//互斥锁用来保证一段时间内只有一个线程在执行一段代码
std::mutex m_state;
std::mutex i_buf;
std::mutex m_estimator;

double latest_time;
Eigen::Vector3d tmp_P;
Eigen::Quaterniond tmp_Q;
Eigen::Vector3d tmp_V;
Eigen::Vector3d tmp_Ba;
Eigen::Vector3d tmp_Bg;
Eigen::Vector3d acc_0;
Eigen::Vector3d gyr_0;
bool init_feature = 0;
bool init_imu = 1;
double last_imu_t = 0;

void predict(const sensor_msgs::ImuConstPtr &imu_msg)
{
    double t = imu_msg->header.stamp.toSec();
    if (init_imu) //if语句里的return，使程序跳出if所在的函数，返回到母函数中继续执行。
    {
        latest_time = t;
        init_imu = 0;
        return;
    }
}
//计算当前imu_msg距离上一个imu_msg的时间间隔
    double dt = t - latest_time;
    latest_time = t;

    double dx = imu_msg->linear_acceleration.x;
    double dy = imu_msg->linear_acceleration.y;
    double dz = imu_msg->linear_acceleration.z;
    Eigen::Vector3d linear_acceleration{dx, dy, dz};

    double rx = imu_msg->angular_velocity.x;
    double ry = imu_msg->angular_velocity.y;
    double rz = imu_msg->angular_velocity.z;
    Eigen::Vector3d angular_velocity{rx, ry, rz};

    Eigen::Vector3d un_acc_0 = tmp_Q * (acc_0 - tmp_Ba) - estimator.g;//a理想值=R(a测量值-Ba)-g  w理想值=w测量值-BW

    Eigen::Vector3d un_gyr = 0.5 * (gyr_0 + angular_velocity) - tmp_Bg;//中值法：取k时刻和k+1时刻的角速度再除2
    tmp_Q = tmp_Q * Utility::deltaQ(un_gyr * dt);//旋转的更新

    Eigen::Vector3d un_acc_1 = tmp_Q * (linear_acceleration - tmp_Ba) - estimator.g;

    Eigen::Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);

    tmp_P = tmp_P + dt * tmp_V + 0.5 * dt * dt * un_acc;
    tmp_V = tmp_V + dt * un_acc;

    acc_0 = linear_acceleration;//迭代赋值
    gyr_0 = angular_velocity;
}
//得到窗口最后一个图像帧的imu项[P,Q,V,ba,bg,a,g]，对imu_buf中剩余imu_msg进行PVQ递推
void update()
{
    TicToc t_predict;
    latest_time = current_time;
    tmp_P = estimator.Ps[WINDOW_SIZE];//vector3d类型
    tmp_Q = estimator.Rs[WINDOW_SIZE];//vector3d类型
    tmp_V = estimator.Vs[WINDOW_SIZE];//vector3d类型
    tmp_Ba = estimator.Bas[WINDOW_SIZE];//vector3d类型
    tmp_Bg = estimator.Bgs[WINDOW_SIZE];//vector3d类型
    acc_0 = estimator.acc_0;
    gyr_0 = estimator.gyr_0;

    queue<sensor_msgs::ImuConstPtr> tmp_imu_buf = imu_buf;
    for (sensor_msgs::ImuConstPtr tmp_imu_msg; !tmp_imu_buf.empty(); tmp_imu_buf.pop())//pop是出队，队列先进先出
        predict(tmp_imu_buf.front());//front指队首指针

}
//getMeasurements函数主要工作就是将接收到的IMU数据和相关的图像数据进行“时间上的对齐”，
// 并将对齐的IMU数据和图像数据进行组合供后边IMU和图像数据处理的时候使用。
// 由于IMU的更新频率比相机要高出好多，所以和一帧图像时间上“对齐”的IMU数据是一个vector类型的容器中存放的IMU数据

//sensor_msgs::PointCloudConstPtr 表示某一帧图像的feature_points
//std::vector<sensor_msgs::ImuConstPtr> 表示当前帧和上一帧时间间隔中的所有IMU数据
std::vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>, sensor_msgs::PointCloudConstPtr>>
getMeasurements()
{
    std::vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>, sensor_msgs::PointCloudConstPtr>> measurements;

    while (true)
    {
        //imu信息接收到后会在imu_callback回调中存入imu_buf，feature消息收到后会在feature_callback中存入feature_buf中
        if (imu_buf.empty() || feature_buf.empty())
            return measurements;
        //下面两个if判断都是对imu数据和image数据在时间戳上的判断（时间要对齐才能进行组合）
        //imu_buf中最后一个imu消息的时间戳比feature_buf中第一个feature消息的时间戳还要小，说明imu数据发出来太早了
        if (!(imu_buf.back()->header.stamp.toSec() > feature_buf.front()->header.stamp.toSec() + estimator.td))
        {
            //ROS_WARN("wait for imu, only should happen at the beginning");
            //imu数据比图像数据要早，所以返回，等待时间上和图像对齐的imu数据到来  为什么不直接把这些imu的数据从imu_buf里丢掉了？这里等，这些数据还在imu_buf里啊
            sum_of_wait++;
            return measurements;
        }
        //imu_buf中第一个数据的时间戳比feature_buf的第一个数据的时间戳还要大
        if (!(imu_buf.front()->header.stamp.toSec() < feature_buf.front()->header.stamp.toSec() + estimator.td))
        {
            ROS_WARN("throw img, only should happen at the beginning");
            //imu数据比图像数据滞后，所以图像数据要出队列
            feature_buf.pop();
            continue;
        }
        //获取队列头部的图像数据，也就是最早发布的图像数据
        sensor_msgs::PointCloudConstPtr img_msg = feature_buf.front();
        feature_buf.pop();

        std::vector<sensor_msgs::ImuConstPtr> IMUs;//两帧图像之间有很多imu，所以用vector
        //imu_buf的最前边的数据时间戳小于图像数据的时间戳的话，就将imu_buf中最前边的数据存入IMUs当中
        while (imu_buf.front()->header.stamp.toSec() < img_msg->header.stamp.toSec() + estimator.td)
        {
            //将imu_buf中最上边的元素插入到IMUs的尾部
            IMUs.emplace_back(imu_buf.front());
            imu_buf.pop();
        }
        IMUs.emplace_back(imu_buf.front());//这里没有pop?
        if (IMUs.empty())
            ROS_WARN("no imu between two image");
        //将和该img_msg时间上“对齐”的imu数据连同该img_msg放入measurements的尾部
        //所以这里的measurements中存放的就是在时间上“对齐”的IMU数据和图像数据
        measurements.emplace_back(IMUs, img_msg);
    }
    return measurements;
}

void imu_callback(const sensor_msgs::ImuConstPtr &imu_msg)
{
    if (imu_msg->header.stamp.toSec() <= last_imu_t)//初值last_imu_t=0
    {
        ROS_WARN("imu message in disorder!");
        return;
    }
    last_imu_t = imu_msg->header.stamp.toSec();

    m_buf.lock();//是个锁，但不知道具体表示啥意思
    imu_buf.push(imu_msg);//将IMU数据保存到imu_buf中，同时执行con.notify_one();唤醒作用于process线程中的获取观测值数据的函数
    m_buf.unlock();
    con.notify_one();

    last_imu_t = imu_msg->header.stamp.toSec();
//这个括弧是啥写法？
    {
        std::lock_guard<std::mutex> lg(m_state);
        predict(imu_msg);
        std_msgs::Header header = imu_msg->header;
        header.frame_id = "world";
        if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR)
            pubLatestOdometry(tmp_P, tmp_Q, tmp_V, header);
    }
}


void feature_callback(const sensor_msgs::PointCloudConstPtr &feature_msg)
{
    if (!init_feature)
    {
        //skip the first detected feature, which doesn't contain optical flow speed
        init_feature = 1;
        return;
    }
    m_buf.lock();
    feature_buf.push(feature_msg);
    m_buf.unlock();
    con.notify_one();
}

void restart_callback(const std_msgs::BoolConstPtr &restart_msg)
{
    if (restart_msg->data == true)
    {
        ROS_WARN("restart the estimator!");
        m_buf.lock();
        while(!feature_buf.empty())
            feature_buf.pop();
        while(!imu_buf.empty())
            imu_buf.pop();
        m_buf.unlock();
        m_estimator.lock();
        estimator.clearState();
        estimator.setParameter();
        m_estimator.unlock();
        current_time = -1;
        last_imu_t = 0;
    }
    return;
}

void relocalization_callback(const sensor_msgs::PointCloudConstPtr &points_msg)
{
    //printf("relocalization callback! \n");
    m_buf.lock();
    relo_buf.push(points_msg);
    m_buf.unlock();
}

// thread: visual-inertial odometry
void process()
{
    while (true)
    {
        std::vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>, sensor_msgs::PointCloudConstPtr>> measurements;
        std::unique_lock<std::mutex> lk(m_buf);
        //只有当 getMeasurements()).size() = 0时wait() 阻塞当前线程，并且在收到imucallback,imagecallback的通知后getMeasurements()).size() != 0即有数据时时才会被解除阻塞
        con.wait(lk, [&]
                 {
            return (measurements = getMeasurements()).size() != 0;
                 });
        // lambda函数的语法定义：（采用了追踪返回类型的方式声明其返回值)
// [capture](parameters) mutable -> return-type{statement;}
// []，捕捉列表，捕捉上下文中的变量以供lambda函数使用
/*[]    表示    空捕捉
 *[=]   表示值  传递方式捕捉所有父作用域的变量(包括this)
 *[&]   表示引用传递方式捕捉所有父作用域的变量(包括this)
 *[var] 表示值  传递方式捕捉变量var
 *[&var]表示引用传递方式捕捉变量var
 *[this]表示值  传递方式捕捉当前this指针 (this。函数体内可以使用Lambda所在类中的成员变量。)
 *其它组合形式：
 * */
        lk.unlock();
        m_estimator.lock();
        //遍历measurements，其实就是遍历获取每一个img_msg和其对应的imu_msg对数据进行处理
        for (auto &measurement : measurements)
        {
            auto img_msg = measurement.second;
            double dx = 0, dy = 0, dz = 0, rx = 0, ry = 0, rz = 0;
            //遍历和当前img_msg时间上“对齐”的IMU数据
            for (auto &imu_msg : measurement.first)
            {
                double t = imu_msg->header.stamp.toSec();
                double img_t = img_msg->header.stamp.toSec() + estimator.td;
                //由于一帧image图像消息对应多个IMU消息，所以针对每一个IMU消息的时间戳，
                // 需要判断IMU的时间戳和image消息的时间戳之间的大小关系，也就是需要判断IMU消息早于image消息先收到，
                // 还是IMU消息晚于image消息收到。
                if (t <= img_t)//imu的数据比图像数据早到
                {
                    //第一次的时候current_time值为-1
                    if (current_time < 0)
                        current_time = t;//imu的采样时间
                    double dt = t - current_time;
                    ROS_ASSERT(dt >= 0);
                    current_time = t;
                    //dx,dy,dz分别是IMU在三个轴方向上的线加速度
                    dx = imu_msg->linear_acceleration.x;
                    dy = imu_msg->linear_acceleration.y;
                    dz = imu_msg->linear_acceleration.z;
                    //rx,ry,rz分别是IMU在三个轴方向上的角速度
                    rx = imu_msg->angular_velocity.x;
                    ry = imu_msg->angular_velocity.y;
                    rz = imu_msg->angular_velocity.z;
                    //对每一个IMU值进行预积分
                    estimator.processIMU(dt, Vector3d(dx, dy, dz), Vector3d(rx, ry, rz));
                    //printf("imu: dt:%f a: %f %f %f w: %f %f %f\n",dt, dx, dy, dz, rx, ry, rz);

                }
                else//图像数据比IMU数据早到,,为什么要这么分啊？
                {
                    double dt_1 = img_t - current_time;
                    double dt_2 = t - img_t;
                    current_time = img_t;
                    ROS_ASSERT(dt_1 >= 0);
                    ROS_ASSERT(dt_2 >= 0);
                    ROS_ASSERT(dt_1 + dt_2 > 0);
                    double w1 = dt_2 / (dt_1 + dt_2);
                    double w2 = dt_1 / (dt_1 + dt_2);
                    dx = w1 * dx + w2 * imu_msg->linear_acceleration.x;
                    dy = w1 * dy + w2 * imu_msg->linear_acceleration.y;
                    dz = w1 * dz + w2 * imu_msg->linear_acceleration.z;
                    rx = w1 * rx + w2 * imu_msg->angular_velocity.x;
                    ry = w1 * ry + w2 * imu_msg->angular_velocity.y;
                    rz = w1 * rz + w2 * imu_msg->angular_velocity.z;
                    //对每一个IMU值进行预积分
                    estimator.processIMU(dt_1, Vector3d(dx, dy, dz), Vector3d(rx, ry, rz));
                    //printf("dimu: dt:%f a: %f %f %f w: %f %f %f\n",dt_1, dx, dy, dz, rx, ry, rz);
                }
            }
            // set relocalization frame
            sensor_msgs::PointCloudConstPtr relo_msg = NULL;//去看传感器数据的definition
            while (!relo_buf.empty())
            {
                relo_msg = relo_buf.front();
                relo_buf.pop();
            }
            if (relo_msg != NULL)
            {
                vector<Vector3d> match_points;
                double frame_stamp = relo_msg->header.stamp.toSec();
                //遍历relo_msg中的points特征点
                for (unsigned int i = 0; i < relo_msg->points.size(); i++)
                {
                    Vector3d u_v_id;
                    u_v_id.x() = relo_msg->points[i].x;
                    u_v_id.y() = relo_msg->points[i].y;
                    u_v_id.z() = relo_msg->points[i].z;
                    match_points.push_back(u_v_id);
                }
                //[重定位帧的平移向量T的x,y,z，旋转四元数w,x,y,z和索引值]

                Vector3d relo_t(relo_msg->channels[0].values[0], relo_msg->channels[0].values[1], relo_msg->channels[0].values[2]);//平移
                Quaterniond relo_q(relo_msg->channels[0].values[3], relo_msg->channels[0].values[4], relo_msg->channels[0].values[5], relo_msg->channels[0].values[6]);//四元数表示的旋转
                Matrix3d relo_r = relo_q.toRotationMatrix();
                int frame_index;
                frame_index = relo_msg->channels[0].values[7];//索引
                estimator.setReloFrame(frame_stamp, frame_index, match_points, relo_t, relo_r);
            }

            ROS_DEBUG("processing vision data with stamp %f \n", img_msg->header.stamp.toSec());

            TicToc t_s;
            map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> image;
            //遍历img_msg中的特征点
            for (unsigned int i = 0; i < img_msg->points.size(); i++)
            {
                int v = img_msg->channels[0].values[i] + 0.5;
                int feature_id = v / NUM_OF_CAM;//这是什么操作？
                int camera_id = v % NUM_OF_CAM;
                //获取img_msg中第i个点的x,y,z坐标，这个是归一化后的坐标值
                double x = img_msg->points[i].x;
                double y = img_msg->points[i].y;
                double z = img_msg->points[i].z;
                //获取像素的坐标值
                double p_u = img_msg->channels[1].values[i];
                double p_v = img_msg->channels[2].values[i];
                //获取像素点在x,y方向上的速度
                double velocity_x = img_msg->channels[3].values[i];
                double velocity_y = img_msg->channels[4].values[i];
                ROS_ASSERT(z == 1);
                Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
                xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y;
                image[feature_id].emplace_back(camera_id,  xyz_uv_velocity);//原地构造，不需要调用构造函数和拷贝构造函数
            }
            estimator.processImage(image, img_msg->header);

            double whole_t = t_s.toc();
            printStatistics(estimator, whole_t);
            std_msgs::Header header = img_msg->header;
            header.frame_id = "world";
            //给Rviz发送里程计信息：odometry、path、relo_path这几个topic
            pubOdometry(estimator, header);
            //给Rviz发送关键点位置
            pubKeyPoses(estimator, header);
            //给Rviz发送相机位置
            pubCameraPose(estimator, header);
            //给Rviz发送点云数据
            pubPointCloud(estimator, header);
            //发布tf转换topic
            pubTF(estimator, header);
            //发送关键帧信息
            pubKeyframe(estimator);
            if (relo_msg != NULL)
                pubRelocalization(estimator);
            //ROS_ERROR("end: %f, at %f", img_msg->header.stamp.toSec(), ros::Time::now().toSec());
        }
        m_estimator.unlock();
        m_buf.lock();
        m_state.lock();
        if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR)
            update();
        m_state.unlock();
        m_buf.unlock();
    }
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "vins_estimator");
    ros::NodeHandle n("~");
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);
    readParameters(n);
    estimator.setParameter();
#ifdef EIGEN_DONT_PARALLELIZE
    ROS_DEBUG("EIGEN_DONT_PARALLELIZE");
#endif
    ROS_WARN("waiting for image and imu...");
//注册发布器
    registerPub(n);

    ros::Subscriber sub_imu = n.subscribe(IMU_TOPIC, 2000, imu_callback, ros::TransportHints().tcpNoDelay());
    ros::Subscriber sub_image = n.subscribe("/feature_tracker/feature", 2000, feature_callback);
    ros::Subscriber sub_restart = n.subscribe("/feature_tracker/restart", 2000, restart_callback);
    ros::Subscriber sub_relo_points = n.subscribe("/pose_graph/match_points", 2000, relocalization_callback);

    std::thread measurement_process{process};//创建线程measurement_process对象，一创建线程对象就启动线程 运行process
    ros::spin();

    return 0;
}
