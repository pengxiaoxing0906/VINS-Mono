#include "parameters.h"

std::string IMAGE_TOPIC;
std::string IMU_TOPIC;
std::vector<std::string> CAM_NAMES;
std::string FISHEYE_MASK;
int MAX_CNT;
int MIN_DIST;
int WINDOW_SIZE;
int FREQ;
double F_THRESHOLD;
int SHOW_TRACK;
int STEREO_TRACK;
int EQUALIZE;
int ROW;
int COL;
int FOCAL_LENGTH;
int FISHEYE;
bool PUB_THIS_FRAME;

template <typename T>
T readParam(ros::NodeHandle &n, std::string name)//传递的实参就是n和`config_file`
{
    T ans;
    if (n.getParam(name, ans))//getParam获取参数，name为参数名，ans为获取到的参数值，string类型 参数的返回类型为bool型
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
    std::string config_file;
    config_file = readParam<std::string>(n, "config_file");//ans-->config_file
    std::cout<<"config_file"<<config_file<<endl;//自己加的，为了查看config_file的具体值
    cv::FileStorage fsSettings(config_file, cv::FileStorage::READ);
    if(!fsSettings.isOpened())
    {
        std::cerr << "ERROR: Wrong path to settings" << std::endl;
    }
    std::string VINS_FOLDER_PATH = readParam<std::string>(n, "vins_folder");

    fsSettings["image_topic"] >> IMAGE_TOPIC;
    fsSettings["imu_topic"] >> IMU_TOPIC;
    MAX_CNT = fsSettings["max_cnt"];//tracking的最大特征点数量
    MIN_DIST = fsSettings["min_dist"];//min distance between two features
    ROW = fsSettings["image_height"];
    COL = fsSettings["image_width"];
    FREQ = fsSettings["freq"];//publish tracking result 的频率,默认设置为10hz
    F_THRESHOLD = fsSettings["F_threshold"];//ransac threshold (pixel)
    SHOW_TRACK = fsSettings["show_track"];//发布image topic
    EQUALIZE = fsSettings["equalize"];//if image is too dark or light, trun on equalize to find enough features,默认为1
    FISHEYE = fsSettings["fisheye"];//默认不用fisheye
    if (FISHEYE == 1)
        FISHEYE_MASK = VINS_FOLDER_PATH + "config/fisheye_mask.jpg";
    CAM_NAMES.push_back(config_file);//注意传递string类型的config_file给CAM_NAMES代表那这个camera有param里的一切参数？

    WINDOW_SIZE = 20;
    STEREO_TRACK = false;//如果双目 把这个参数改为true
    FOCAL_LENGTH = 460;
    PUB_THIS_FRAME = false;

    if (FREQ == 0)
        FREQ = 100;

    fsSettings.release();


}
