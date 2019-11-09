#include "parameters.h"

double INIT_DEPTH;
double MIN_PARALLAX;
double ACC_N, ACC_W;
double GYR_N, GYR_W;

std::vector<Eigen::Matrix3d> RIC;
std::vector<Eigen::Vector3d> TIC;

Eigen::Vector3d G{0.0, 0.0, 9.8};

double BIAS_ACC_THRESHOLD;
double BIAS_GYR_THRESHOLD;
double SOLVER_TIME;
int NUM_ITERATIONS;
int ESTIMATE_EXTRINSIC;
int ESTIMATE_TD;
int ROLLING_SHUTTER;
std::string EX_CALIB_RESULT_PATH;
std::string VINS_RESULT_PATH;
std::string IMU_TOPIC;
double ROW, COL;
double TD, TR;

template <typename T>
T readParam(ros::NodeHandle &n, std::string name)
{
    T ans;
    if (n.getParam(name, ans))
    {
        ROS_INFO_STREAM("Loaded " << name << ": " << ans);
    }
    else
    {
        ROS_ERROR_STREAM("Failed to load " << name);
        n.shutdown();
    }
    return ans;
}

void readParameters(ros::NodeHandle &n)
{
    //[1]读取参数文件euroc_config.yaml
    std::string config_file;
    config_file = readParam<std::string>(n, "config_file");
    cv::FileStorage fsSettings(config_file, cv::FileStorage::READ);

    //[2]文件是否正确打开判断
    if(!fsSettings.isOpened())
    {
        std::cerr << "ERROR: Wrong path to settings" << std::endl;
    }

    //[3]读取相应参数
    fsSettings["imu_topic"] >> IMU_TOPIC;//imu_topic: "/imu0"

    SOLVER_TIME = fsSettings["max_solver_time"];//solver的最大迭代时间0.04ms
    NUM_ITERATIONS = fsSettings["max_num_iterations"];//最大迭代次数8
    MIN_PARALLAX = fsSettings["keyframe_parallax"];//最小视差10pixel
    MIN_PARALLAX = MIN_PARALLAX / FOCAL_LENGTH;//最小视差=最小视差/焦距=10.0/460.0

    //[4]设置输出路径
    std::string OUTPUT_PATH;
    fsSettings["output_path"] >> OUTPUT_PATH;
    VINS_RESULT_PATH = OUTPUT_PATH + "/vins_result_no_loop.csv";
    std::cout << "result path " << VINS_RESULT_PATH << std::endl;

    FileSystemHelper::createDirectoryIfNotExists(OUTPUT_PATH.c_str());//如果文件不存在，就创建一个

    std::ofstream fout(VINS_RESULT_PATH, std::ios::out);
    fout.close();

    //【5】继续读取参数
    ACC_N = fsSettings["acc_n"];//加速度计的noise
    ACC_W = fsSettings["acc_w"];//加速度计的bias
    GYR_N = fsSettings["gyr_n"];//陀螺仪的noise
    GYR_W = fsSettings["gyr_w"];//陀螺仪的bias
    G.z() = fsSettings["g_norm"];//重力
    ROW = fsSettings["image_height"];
    COL = fsSettings["image_width"];
    ROS_INFO("ROW: %f COL: %f ", ROW, COL);

    //【6】读取imu和camera的外参，根据读取的参数执行相应的操作
    ESTIMATE_EXTRINSIC = fsSettings["estimate_extrinsic"];
    if (ESTIMATE_EXTRINSIC == 2)//表示完全让算法标定imu和相机之间的外参
    {
        ROS_WARN("have no prior about extrinsic param, calibrate extrinsic param");
        RIC.push_back(Eigen::Matrix3d::Identity());//给RIC赋值为单位阵
        TIC.push_back(Eigen::Vector3d::Zero());//给TIC赋值为另矩阵
        EX_CALIB_RESULT_PATH = OUTPUT_PATH + "/extrinsic_parameter.csv";

    }
    else 
    {
        if ( ESTIMATE_EXTRINSIC == 1)//表示我们提供一个初值，算法仅对初值进行优化
        {
            ROS_WARN(" Optimize extrinsic param around initial guess!");
            EX_CALIB_RESULT_PATH = OUTPUT_PATH + "/extrinsic_parameter.csv";
        }
        if (ESTIMATE_EXTRINSIC == 0)//表示算法信任我们提供的外参，不对其做任何修改
            ROS_WARN(" fix extrinsic param ");

        cv::Mat cv_R, cv_T;
        fsSettings["extrinsicRotation"] >> cv_R;//imu与camera的旋转外参
        fsSettings["extrinsicTranslation"] >> cv_T;//imu与camera的平移外参
        Eigen::Matrix3d eigen_R;
        Eigen::Vector3d eigen_T;
        cv::cv2eigen(cv_R, eigen_R);//opencv的数据结构转eigen
        cv::cv2eigen(cv_T, eigen_T);
        Eigen::Quaterniond Q(eigen_R);
        eigen_R = Q.normalized();//四元数归一化咋就转成Matrix3d啦？
        RIC.push_back(eigen_R);
        TIC.push_back(eigen_T);
        ROS_INFO_STREAM("Extrinsic_R : " << std::endl << RIC[0]);
        ROS_INFO_STREAM("Extrinsic_T : " << std::endl << TIC[0].transpose());
        
    } 

    //[7]给参数赋初始值
    INIT_DEPTH = 5.0;
    BIAS_ACC_THRESHOLD = 0.1;
    BIAS_GYR_THRESHOLD = 0.1;

    //[8]继续读取参数TD，图像和IMU数据在时间上的偏移量，这里配置文件中为0.0
    TD = fsSettings["td"];//0 initial value of time offset. unit: s. readed image clock + td = real image clock (IMU clock)
    ESTIMATE_TD = fsSettings["estimate_td"];//0
    if (ESTIMATE_TD)
        ROS_INFO_STREAM("Unsynchronized sensors, online estimate time offset, initial td: " << TD);
    else
        ROS_INFO_STREAM("Synchronized sensors, fix time offset: " << TD);

    ROLLING_SHUTTER = fsSettings["rolling_shutter"];//0是全局快门，1是卷帘快门
    if (ROLLING_SHUTTER)
    {
        TR = fsSettings["rolling_shutter_tr"];//这个时间默认设置的是0
        ROS_INFO_STREAM("rolling shutter camera, read out time per line: " << TR);
    }
    else
    {
        TR = 0;
    }
    
    fsSettings.release();
}
