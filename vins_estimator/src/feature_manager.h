#ifndef FEATURE_MANAGER_H
#define FEATURE_MANAGER_H

#include <list>
#include <algorithm>
#include <vector>
#include <numeric>
using namespace std;

#include <eigen3/Eigen/Dense>
using namespace Eigen;

#include <ros/console.h>
#include <ros/assert.h>

#include "parameters.h"

class FeaturePerFrame //一个特征点的属性:空间位置，像素位置，特征点的跟踪速度
{
  public:
    FeaturePerFrame(const Eigen::Matrix<double, 7, 1> &_point, double td)
    {
        point.x() = _point(0);
        point.y() = _point(1);
        point.z() = _point(2);
        uv.x() = _point(3);
        uv.y() = _point(4);
        velocity.x() = _point(5); 
        velocity.y() = _point(6); 
        cur_td = td;
    }
    double cur_td;
    Vector3d point;
    Vector2d uv;
    Vector2d velocity;
    double z;
    bool is_used;
    double parallax;//视差
    MatrixXd A;
    VectorXd b;
    double dep_gradient;//?
};

class FeaturePerId//每个路标点由多个连续的图像观测到 FeaturePerFrame1,FeaturePerFrame2,FeaturePerFrame3,
{
  public:
    const int feature_id;//特征点的id
    int start_frame;//第一次出现某特征点的帧号
    vector<FeaturePerFrame> feature_per_frame;//管理对应帧的属性

    int used_num;//出现的次数
    bool is_outlier;
    bool is_margin;
    double estimated_depth;//估计的深度
    int solve_flag; // 0 haven't solve yet; 1 solve succ; 2 solve fail;

    Vector3d gt_p;

    FeaturePerId(int _feature_id, int _start_frame)
        : feature_id(_feature_id), start_frame(_start_frame),
          used_num(0), estimated_depth(-1.0), solve_flag(0)
    {
    }

    int endFrame();//得到该特征点最后一次跟踪到的帧号
};

class FeatureManager//滑窗内所有的路标点
{
  public:
    FeatureManager(Matrix3d _Rs[]);

    void setRic(Matrix3d _ric[]);//imu和相机的旋转外参

    void clearState();

    int getFeatureCount();//计算滑窗内被track过的特征点的数量
//parallax是视差的意思
    bool addFeatureCheckParallax(int frame_count, const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image, double td);
    void debugShow();
    vector<pair<Vector3d, Vector3d>> getCorresponding(int frame_count_l, int frame_count_r);

    //void updateDepth(const VectorXd &x);
    void setDepth(const VectorXd &x);
    void removeFailures();
    void clearDepth(const VectorXd &x);
    VectorXd getDepthVector();
    void triangulate(Vector3d Ps[], Vector3d tic[], Matrix3d ric[]);
    void removeBackShiftDepth(Eigen::Matrix3d marg_R, Eigen::Vector3d marg_P, Eigen::Matrix3d new_R, Eigen::Vector3d new_P);
    void removeBack();
    void removeFront(int frame_count);
    void removeOutlier();
    list<FeaturePerId> feature;
    int last_track_num;
//理解这三个类花了点时间：终于看明白了，参考：https://blog.csdn.net/liuzheng1/article/details/90052050
  private:
    double compensatedParallax2(const FeaturePerId &it_per_id, int frame_count);//视差补偿
    const Matrix3d *Rs;
    Matrix3d ric[NUM_OF_CAM];
};

#endif