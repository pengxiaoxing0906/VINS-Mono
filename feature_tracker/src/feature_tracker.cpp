#include "feature_tracker.h"

int FeatureTracker::n_id = 0;

bool inBorder(const cv::Point2f &pt)//判断点是否在图像内
{
    const int BORDER_SIZE = 1;
    int img_x = cvRound(pt.x);
    int img_y = cvRound(pt.y);
    return BORDER_SIZE <= img_x && img_x < COL - BORDER_SIZE && BORDER_SIZE <= img_y && img_y < ROW - BORDER_SIZE;
}

void reduceVector(vector<cv::Point2f> &v, vector<uchar> status)//根据状态status进行重组，将staus中为1的对应点保存下来，为0的对应点对去除掉
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}

void reduceVector(vector<int> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}


FeatureTracker::FeatureTracker()
{
}

//setMask()作用：对跟踪到的特征点对依据跟踪次数进行从多到少的排序，并放到原集合中
//在已跟踪到角点的位置上，将mask对应位置上设为0，意为在cv::goodFeaturesToTrack(forw_img, n_pts, MAX_CNT - forw_pts.size(), 0.01, MIN_DIST, mask);进行操作时在该点不再重复进行角点检测，这样可以使角点分布更加均匀。

void FeatureTracker::setMask()
{
    //[1]判断是否是鱼眼相机，采取相应的掩膜方式
    if(FISHEYE)
        mask = fisheye_mask.clone();
    else
        mask = cv::Mat(ROW, COL, CV_8UC1, cv::Scalar(255));
    
    //[2]定义复合数据类型，三个数据类型分别表示特征点被track次数、特征点、特征点的id
    vector<pair<int, pair<cv::Point2f, int>>> cnt_pts_id;

   //[3]将当前帧中的特征点，及特征点id和被track的次数打包push进cn_pts_id中
    for (unsigned int i = 0; i < forw_pts.size(); i++)
        cnt_pts_id.push_back(make_pair(track_cnt[i], make_pair(forw_pts[i], ids[i])));

    //[4]按特征点的跟踪次数由大到小排序
    sort(cnt_pts_id.begin(), cnt_pts_id.end(), [](const pair<int, pair<cv::Point2f, int>> &a, const pair<int, pair<cv::Point2f, int>> &b)
         {
            return a.first > b.first;
         });

    //[5]清空track_cnt，forw_pts，id并重新存入
    forw_pts.clear();
    ids.clear();
    track_cnt.clear();

    //[6]遍历cnt_pts_id，操作目的为使图像提取的特征点更均匀
    for (auto &it : cnt_pts_id)
    {
        if (mask.at<uchar>(it.second.first) == 255) //当前特征点位置对应的mask值为255，则保留当前特征点，将对应的特征点位置pts，id，被追踪次数cnt分别存入
        {
            forw_pts.push_back(it.second.first);
            ids.push_back(it.second.second);
            track_cnt.push_back(it.first);
            //图片，点，半径，颜色为0表示在角点检测在该点不起作用,粗细（-1）表示填充
            //在mask中将当前特征点周围半径为MIN_DIST的区域设置为0，后面不再选取该区域内的点（使跟踪点不集中在一个区域上）
            cv::circle(mask, it.second.first, MIN_DIST, 0, -1);
        }
    }
}

void FeatureTracker::addPoints()//把新提取的特征点放到forw_pts中，id初始化-1,track_cnt初始化为1.
{
    for (auto &p : n_pts)
    {
        forw_pts.push_back(p);
        ids.push_back(-1);
        track_cnt.push_back(1);
    }
}

void FeatureTracker::readImage(const cv::Mat &_img, double _cur_time)
{
    cv::Mat img;
    TicToc t_r;
    cur_time = _cur_time;

    //【1】判断图像是否需要均衡化
    if (EQUALIZE)//均衡化增强对比度 默认值为1,表示图像太亮或者太暗
    {
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));//opencv里头的函数
        TicToc t_c;
        clahe->apply(_img, img);
        ROS_DEBUG("CLAHE costs: %fms", t_c.toc());
    }
    else
        img = _img;

   //[2]如果当前帧的图像数据forw_img为空，说明当前是第一次读入图像数据,将读入的图像赋给当前帧forw_img,同时，还将读入的图像赋给prev_img、cur_img
    if (forw_img.empty())
    {
        prev_img = cur_img = forw_img = img;
    }
    else
    {
        forw_img = img;
    }

    forw_pts.clear();//此时forw_pts还保存的是上一帧图像中的特征点，所以把它清除

    //[3]判断光流跟踪前一帧中特征点规模是否不为0，不为0表示有图像数据点，对其进行光流跟踪
    if (cur_pts.size() > 0)
    {
        TicToc t_o;
        vector<uchar> status;//status表示 cur_pts和forw_pts中对应点对是否跟踪成功,无法被追踪到的点标记为0
        vector<float> err;
        cv::calcOpticalFlowPyrLK(cur_img, forw_img, cur_pts, forw_pts, status, err, cv::Size(21, 21), 3);//calcOpticalFlowPyrLK() 从cur_pts到forw_pts做LK金字塔光流法

        for (int i = 0; i < int(forw_pts.size()); i++)//光流跟踪结束后，判断跟踪成功的角点是否都在图像内,不在图像内的status置为0
            if (status[i] && !inBorder(forw_pts[i]))
                status[i] = 0;

        //根据status,把跟踪失败的点剔除
        //不仅要从当前帧数据forw_pts中剔除，而且还要从cur_un_pts、prev_pts和cur_pts中剔除
        //prev_pts和cur_pts中的特征点是一一对应的
        //记录特征点id的ids，和记录特征点被跟踪次数的track_cnt也要剔除
        reduceVector(prev_pts, status);
        reduceVector(cur_pts, status);
        reduceVector(forw_pts, status);
        reduceVector(ids, status);
        reduceVector(cur_un_pts, status);
        reduceVector(track_cnt, status);
        ROS_DEBUG("temporal optical flow costs: %fms", t_o.toc());
    }

   //[4]光流追踪成功,特征点被成功跟踪的次数track_cnt就加1
    for (auto &n : track_cnt)//track_cnt为每个角点的跟踪次数
        n++;

    //[5]发布这一帧数据
    if (PUB_THIS_FRAME)
    {
        rejectWithF();//通过基本矩阵F剔除outliers

        ROS_DEBUG("set mask begins");
        TicToc t_m;
        setMask();//保证相邻的特征点之间要相隔30个像素,设置mask
        ROS_DEBUG("set mask costs %fms", t_m.toc());

        ROS_DEBUG("detect feature begins");
        TicToc t_t;

        //[6] //计算是否需要提取新的特征点
        int n_max_cnt = MAX_CNT - static_cast<int>(forw_pts.size());
        if (n_max_cnt > 0)
        {
            if(mask.empty())
                cout << "mask is empty " << endl;
            if (mask.type() != CV_8UC1)
                cout << "mask type wrong " << endl;
            if (mask.size() != forw_img.size())
                cout << "wrong size " << endl;

            /**
    * cv::goodFeaturesToTrack()
    * @brief   在mask中不为0的区域检测新的特征点
    * @optional    ref:https://docs.opencv.org/3.1.0/dd/d1a/group__imgproc__feature.html#ga1d6bb77486c8f92d79c8793ad995d541
    * @param[in]    InputArray _image=forw_img 输入图像
    * @param[out]   _corners=n_pts 存放检测到的角点的vector
    * @param[in]    maxCorners=MAX_CNT - forw_pts.size() 返回的角点的数量的最大值
    * @param[in]    qualityLevel=0.01 角点质量水平的最低阈值（范围为0到1，质量最高角点的水平为1），小于该阈值的角点被拒绝
    * @param[in]    minDistance=MIN_DIST 返回角点之间欧式距离的最小值
    * @param[in]    _mask=mask 和输入图像具有相同大小，类型必须为CV_8UC1,用来描述图像中感兴趣的区域，只在感兴趣区域中检测角点
    * @param[in]    blockSize：计算协方差矩阵时的窗口大小
    * @param[in]    useHarrisDetector：指示是否使用Harris角点检测，如不指定，则计算shi-tomasi角点
    * @param[in]    harrisK：Harris角点检测需要的k值
    * @return      void
    */


            cv::goodFeaturesToTrack(forw_img, n_pts, MAX_CNT - forw_pts.size(), 0.01, MIN_DIST, mask);//MAX_CNT=150
        }
        else
            n_pts.clear();//n_pts是用来存储提取的新的特征点的，在这个条件下表示不需要提取新的特征点，将上一帧中提取的点clear掉
        ROS_DEBUG("detect feature costs: %fms", t_t.toc());

        //[7]将n_pts中新提取到的角点放到forw_pts中，id初始化-1,track_cnt初始化为1.
        ROS_DEBUG("add feature begins");
        TicToc t_a;
        addPoints();
        ROS_DEBUG("selectFeature costs: %fms", t_a.toc());
    }

   //[8]将当前帧数据传递给上一帧
    prev_img = cur_img;//在第一帧处理中还是等于当前帧forw_img
    prev_pts = cur_pts;//在第一帧中不做处理
    prev_un_pts = cur_un_pts;//在第一帧中不做处理
    cur_img = forw_img;//将当前帧赋值给上一帧
    cur_pts = forw_pts;

    //[9]//从第二张图像输入后每进行一次循环，最后还需要对匹配的特征点对进行畸变矫正和深度归一化，计算速度
    undistortedPoints();
    prev_time = cur_time;
}

void FeatureTracker::rejectWithF()
{
    if (forw_pts.size() >= 8)
    {
        ROS_DEBUG("FM ransac begins");
        TicToc t_f;
        //[1]构建cv::Point2f类型的点为cv::findFundamentalMat()做准备
        vector<cv::Point2f> un_cur_pts(cur_pts.size()), un_forw_pts(forw_pts.size());
        for (unsigned int i = 0; i < cur_pts.size(); i++)
        {
            //根据不同的相机模型将二维坐标转换到三维坐标
            //对于PINHOLE（针孔相机）可将像素坐标转换到归一化平面并去畸变
            //对于CATA（卡特鱼眼相机）将像素坐标投影到单位圆内
            Eigen::Vector3d tmp_p;
            m_camera->liftProjective(Eigen::Vector2d(cur_pts[i].x, cur_pts[i].y), tmp_p);
            tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + COL / 2.0;//  转换为像素坐标
            tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + ROW / 2.0;
            un_cur_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());

            m_camera->liftProjective(Eigen::Vector2d(forw_pts[i].x, forw_pts[i].y), tmp_p);
            tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + COL / 2.0;
            tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + ROW / 2.0;
            un_forw_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());
        }

        vector<uchar> status;
        //[2]调用cv::findFundamentalMat对un_cur_pts和un_forw_pts计算F矩阵,RANSAC去除outliers
        cv::findFundamentalMat(un_cur_pts, un_forw_pts, cv::FM_RANSAC, F_THRESHOLD, 0.99, status);
        int size_a = cur_pts.size();
        reduceVector(prev_pts, status);//根据status进行重组
        reduceVector(cur_pts, status);
        reduceVector(forw_pts, status);
        reduceVector(cur_un_pts, status);
        reduceVector(ids, status);
        reduceVector(track_cnt, status);
        ROS_DEBUG("FM ransac: %d -> %lu: %f", size_a, forw_pts.size(), 1.0 * forw_pts.size() / size_a);
        ROS_DEBUG("FM ransac costs: %fms", t_f.toc());
    }
}

bool FeatureTracker::updateID(unsigned int i)
{
    if (i < ids.size())
    {
        if (ids[i] == -1)
            ids[i] = n_id++;
        return true;
    }
    else
        return false;
}

void FeatureTracker::readIntrinsicParameter(const string &calib_file)
{
    ROS_INFO("reading paramerter of camera %s", calib_file.c_str());
    m_camera = CameraFactory::instance()->generateCameraFromYamlFile(calib_file);
}

void FeatureTracker::showUndistortion(const string &name)
{
    cv::Mat undistortedImg(ROW + 600, COL + 600, CV_8UC1, cv::Scalar(0));
    vector<Eigen::Vector2d> distortedp, undistortedp;
    for (int i = 0; i < COL; i++)
        for (int j = 0; j < ROW; j++)
        {
            Eigen::Vector2d a(i, j);
            Eigen::Vector3d b;
            m_camera->liftProjective(a, b);
            distortedp.push_back(a);
            undistortedp.push_back(Eigen::Vector2d(b.x() / b.z(), b.y() / b.z()));
            //printf("%f,%f->%f,%f,%f\n)\n", a.x(), a.y(), b.x(), b.y(), b.z());
        }
    for (int i = 0; i < int(undistortedp.size()); i++)
    {
        cv::Mat pp(3, 1, CV_32FC1);
        pp.at<float>(0, 0) = undistortedp[i].x() * FOCAL_LENGTH + COL / 2;
        pp.at<float>(1, 0) = undistortedp[i].y() * FOCAL_LENGTH + ROW / 2;
        pp.at<float>(2, 0) = 1.0;
        //cout << trackerData[0].K << endl;
        //printf("%lf %lf\n", p.at<float>(1, 0), p.at<float>(0, 0));
        //printf("%lf %lf\n", pp.at<float>(1, 0), pp.at<float>(0, 0));
        if (pp.at<float>(1, 0) + 300 >= 0 && pp.at<float>(1, 0) + 300 < ROW + 600 && pp.at<float>(0, 0) + 300 >= 0 && pp.at<float>(0, 0) + 300 < COL + 600)
        {
            undistortedImg.at<uchar>(pp.at<float>(1, 0) + 300, pp.at<float>(0, 0) + 300) = cur_img.at<uchar>(distortedp[i].y(), distortedp[i].x());
        }
        else
        {
            //ROS_ERROR("(%f %f) -> (%f %f)", distortedp[i].y, distortedp[i].x, pp.at<float>(1, 0), pp.at<float>(0, 0));
        }
    }
    cv::imshow(name, undistortedImg);
    cv::waitKey(0);
}

void FeatureTracker::undistortedPoints()//计算点的在x,y方向上的跟踪速度
{
    //【1】清除cur_un_pts、cur_un_pts_map的状态
    cur_un_pts.clear();
    cur_un_pts_map.clear();
    //cv::undistortPoints(cur_pts, un_pts, K, cv::Mat());

    //[2]给cur_un_pts、cur_un_pts_map装进新的值
    for (unsigned int i = 0; i < cur_pts.size(); i++)
    {
        Eigen::Vector2d a(cur_pts[i].x, cur_pts[i].y);
        Eigen::Vector3d b;
        m_camera->liftProjective(a, b);//转像素坐标
        cur_un_pts.push_back(cv::Point2f(b.x() / b.z(), b.y() / b.z()));
        cur_un_pts_map.insert(make_pair(ids[i], cv::Point2f(b.x() / b.z(), b.y() / b.z())));
        //printf("cur pts id %d %f %f", ids[i], cur_un_pts[i].x, cur_un_pts[i].y);
    }

    // 【3】判断上一帧中特征点点是否为0，不为0则计算点跟踪的速度
    if (!prev_un_pts_map.empty())
    {
        double dt = cur_time - prev_time;
        pts_velocity.clear(); // pts_velocity表示当前帧相对前一帧特征点沿x,y方向的像素移动速度
        for (unsigned int i = 0; i < cur_un_pts.size(); i++)
        {
            if (ids[i] != -1)//如果点不是第一次被track
            {
                std::map<int, cv::Point2f>::iterator it;
                it = prev_un_pts_map.find(ids[i]);
                if (it != prev_un_pts_map.end())
                {
                    double v_x = (cur_un_pts[i].x - it->second.x) / dt;//计算x方向上的速度
                    double v_y = (cur_un_pts[i].y - it->second.y) / dt;//计算y方向上的速度
                    pts_velocity.push_back(cv::Point2f(v_x, v_y));
                }
                else
                    pts_velocity.push_back(cv::Point2f(0, 0));
            }
            else//点第一次被track
            {
                pts_velocity.push_back(cv::Point2f(0, 0));
            }
        }
    }
    else//为0则push一个值
    {
        for (unsigned int i = 0; i < cur_pts.size(); i++)
        {
            pts_velocity.push_back(cv::Point2f(0, 0));
        }
    }

    //[4]当前帧中的点和点的id传递到上一帧
    prev_un_pts_map = cur_un_pts_map;
}
