#include "initial_ex_rotation.h"

InitialEXRotation::InitialEXRotation(){
    frame_count = 0;
    Rc.push_back(Matrix3d::Identity());
    Rc_g.push_back(Matrix3d::Identity());
    Rimu.push_back(Matrix3d::Identity());
    ric = Matrix3d::Identity();
}

bool InitialEXRotation::CalibrationExRotation(vector<pair<Vector3d, Vector3d>> corres, Quaterniond delta_q_imu, Matrix3d &calib_ric_result)
{
    frame_count++;
    Rc.push_back(solveRelativeR(corres));//求解两帧之间的相对位姿
    Rimu.push_back(delta_q_imu.toRotationMatrix());
    Rc_g.push_back(ric.inverse() * delta_q_imu * ric);//求出来的是相机上一帧到下一帧的旋转，此表达式由公式推导得出，理论上跟Rc相等

    Eigen::MatrixXd A(frame_count * 4, 4);//这一部分需要看理论知识
    A.setZero();
    int sum_ok = 0;
    for (int i = 1; i <= frame_count; i++)
    {
        Quaterniond r1(Rc[i]);
        Quaterniond r2(Rc_g[i]);

        double angular_distance = 180 / M_PI * r1.angularDistance(r2);//求角度误差
        ROS_DEBUG(
            "%d %f", i, angular_distance);

        double huber = angular_distance > 5.0 ? 5.0 / angular_distance : 1.0;//求核函数值
        ++sum_ok;
        Matrix4d L, R;

        double w = Quaterniond(Rc[i]).w();//取四元数的实部！！相机的旋转取左L
        Vector3d q = Quaterniond(Rc[i]).vec();//取四元数的虚部
        L.block<3, 3>(0, 0) = w * Matrix3d::Identity() + Utility::skewSymmetric(q);
        L.block<3, 1>(0, 3) = q;
        L.block<1, 3>(3, 0) = -q.transpose();
        L(3, 3) = w;

        Quaterniond R_ij(Rimu[i]);//imu的旋转取右R
        w = R_ij.w();
        q = R_ij.vec();
        R.block<3, 3>(0, 0) = w * Matrix3d::Identity() - Utility::skewSymmetric(q);
        R.block<3, 1>(0, 3) = q;
        R.block<1, 3>(3, 0) = -q.transpose();
        R(3, 3) = w;

        A.block<4, 4>((i - 1) * 4, 0) = huber * (L - R);//加上huber权重
    }

    JacobiSVD<MatrixXd> svd(A, ComputeFullU | ComputeFullV);//svd分解
    Matrix<double, 4, 1> x = svd.matrixV().col(3);
    Quaterniond estimated_R(x);//上面估计的R为imu到相机的旋转
    ric = estimated_R.toRotationMatrix().inverse();
    //cout << svd.singularValues().transpose() << endl;
    //cout << ric << endl;
    Vector3d ric_cov;
    ric_cov = svd.singularValues().tail<3>();
    if (frame_count >= WINDOW_SIZE && ric_cov(1) > 0.25)
    {
        calib_ric_result = ric;
        return true;
    }
    else
        return false;
}

Matrix3d InitialEXRotation::solveRelativeR(const vector<pair<Vector3d, Vector3d>> &corres)//求两帧之间的旋转及平移
{
    //如果点对数大于等于9，利用对极约束来求解相对运动
    if (corres.size() >= 9)
    {
        //[1]构建cv::Point2f类型的数据
        vector<cv::Point2f> ll, rr;
        for (int i = 0; i < int(corres.size()); i++)
        {
            ll.push_back(cv::Point2f(corres[i].first(0), corres[i].first(1)));//取点的x,y坐标push进vector中
            rr.push_back(cv::Point2f(corres[i].second(0), corres[i].second(1)));
        }

       /*
        * 目前问题汇总
        * 1. sigma=1.0，RANSAC最大迭代次数200
        * 2. corres是谁对谁的旋转 framecount-1,framecount 就是参考帧到当前帧
        * 3. 如何从H矩阵恢复出R和t
        *
        */


        //【2】求解出本质矩阵和H矩阵并分别计算得分选择合适的模型分解求出旋转和平移
        float SE,SH;
        double sigma=1.0;
        Mat K;//相机内参矩阵
        //求解本质矩阵E
        cv::Mat E = cv::findEssentialMat(ll, rr);
        //计算SE的函数
        float SE=CheckEssential(E,sigma);

        //求解单应矩阵H
        cv::Mat H=cv::findHomography(ll,rr,RANSAC,3,noArray(),200,0.99);//200代表RANSAC的最大迭代次数，0.99代表可信度
        //计算SH分数
        float SH=CheckHomography(H,sigma);

        float RH = SH/(SH+SF);//定义ratio
        if(RH>0.40)
        {
            //选择用H矩阵来恢复R,t

           decomposeH(H);//求解出的旋转和平移放在vector<cv::Mat>类型中，vR,vt，共8组
           //找出最合适的R,t
            int bestGood = 0;
            int secondBestGood = 0;
            int bestSolutionIdx = -1;
            float bestParallax = -1;
            vector<cv::Point3f> bestP3D;
            vector<bool> bestTriangulated;

            // Instead of applying the visibility constraints proposed in the Faugeras' paper (which could fail for points seen with low parallax)
            // We reconstruct all hypotheses and check in terms of triangulated points and parallax
            for(size_t i=0; i<8; i++)
            {
                float parallaxi;
                vector<cv::Point3f> vP3Di;
                vector<bool> vbTriangulatedi;
                cv::Mat K=(Mat_<double>(3,3)<<461.6,0,363.0,0,460.3,248.1,0,0,1);
                int nGood = CheckRT(vR[i],vt[i],mvKeys1,mvKeys2,mvMatches12,vbMatchesInliers,K,vP3Di, 4.0*mSigma2, vbTriangulatedi, parallaxi);

                if(nGood>bestGood)
                {
                    secondBestGood = bestGood;
                    bestGood = nGood;
                    bestSolutionIdx = i;
                    bestParallax = parallaxi;
                    bestP3D = vP3Di;
                    bestTriangulated = vbTriangulatedi;
                }
                else if(nGood>secondBestGood)
                {
                    secondBestGood = nGood;
                }
            }


            if(secondBestGood<0.75*bestGood && bestParallax>=minParallax && bestGood>minTriangulated && bestGood>0.9*N)
            {
                vR[bestSolutionIdx].copyTo(R21);
                vt[bestSolutionIdx].copyTo(t21);
                vP3D = bestP3D;
                vbTriangulated = bestTriangulated;

            }

        }
        else
        {
            //选择用基础矩阵E来恢复R,t
            cv::Mat_<double> R1, R2, t1, t2;
            decomposeE(E, R1, R2, t1, t2);

            if (determinant(R1) + 1.0 < 1e-09)
            {
                E = -E;
                decomposeE(E,R1,R2,t1,t2);
            }

            //选出更合适的旋转矩阵
            double ratio1 = max(testTriangulation(ll, rr, R1, t1), testTriangulation(ll, rr, R1, t2));
            double ratio2 = max(testTriangulation(ll, rr, R2, t1), testTriangulation(ll, rr, R2, t2));
            cv::Mat_<double> ans_R_cv = ratio1 > ratio2 ? R1 : R2;


        }


        //【5】之前求的是参考帧到当前帧的变换，求转置等于求逆，即是当前帧到参考帧的变换
        Matrix3d ans_R_eigen;
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                ans_R_eigen(j, i) = ans_R_cv(i, j);
        return ans_R_eigen;
    }
    //如果不满足条件，返回单位矩阵
    return Matrix3d::Identity();
}

double InitialEXRotation::testTriangulation(const vector<cv::Point2f> &l,
                                          const vector<cv::Point2f> &r,
                                          cv::Mat_<double> R, cv::Mat_<double> t)
{
    cv::Mat pointcloud;
    cv::Matx34f P = cv::Matx34f(1, 0, 0, 0,
                                0, 1, 0, 0,
                                0, 0, 1, 0);
    cv::Matx34f P1 = cv::Matx34f(R(0, 0), R(0, 1), R(0, 2), t(0),
                                 R(1, 0), R(1, 1), R(1, 2), t(1),
                                 R(2, 0), R(2, 1), R(2, 2), t(2));
    cv::triangulatePoints(P, P1, l, r, pointcloud);
    int front_count = 0;
    for (int i = 0; i < pointcloud.cols; i++)
    {
        double normal_factor = pointcloud.col(i).at<float>(3);

        cv::Mat_<double> p_3d_l = cv::Mat(P) * (pointcloud.col(i) / normal_factor);
        cv::Mat_<double> p_3d_r = cv::Mat(P1) * (pointcloud.col(i) / normal_factor);
        if (p_3d_l(2) > 0 && p_3d_r(2) > 0)
            front_count++;
    }
    ROS_DEBUG("MotionEstimator: %f", 1.0 * front_count / pointcloud.cols);
    return 1.0 * front_count / pointcloud.cols;
}

void InitialEXRotation::decomposeE(cv::Mat E,
                                 cv::Mat_<double> &R1, cv::Mat_<double> &R2,
                                 cv::Mat_<double> &t1, cv::Mat_<double> &t2)
{
    cv::SVD svd(E, cv::SVD::MODIFY_A);
    cv::Matx33d W(0, -1, 0,
                  1, 0, 0,
                  0, 0, 1);
    cv::Matx33d Wt(0, 1, 0,
                   -1, 0, 0,
                   0, 0, 1);
    R1 = svd.u * cv::Mat(W) * svd.vt;
    R2 = svd.u * cv::Mat(Wt) * svd.vt;
    t1 = svd.u.col(2);
    t2 = -svd.u.col(2);
}
 float InitialEXRotation::CheckEssential(cv::Mat E,float sigma)
 {
    // const int N = mvMatches12.size();
    //输入K或者传入K矩阵
     cv::Mat K=(Mat_<double>(3,3)<<461.6,0,363.0,0,460.3,248.1,0,0,1);
    cv::Mat Kinv=K.inv();
    cv::Mat F=Kinv.transpose()*E*Kinv;

     const float f11 = F.at<float>(0,0);
     const float f12 = F.at<float>(0,1);
     const float f13 = F.at<float>(0,2);
     const float f21 = F.at<float>(1,0);
     const float f22 = F.at<float>(1,1);
     const float f23 = F.at<float>(1,2);
     const float f31 = F.at<float>(2,0);
     const float f32 = F.at<float>(2,1);
     const float f33 = F.at<float>(2,2);

     //vbMatchesInliers.resize(N);

     float score = 0;

     const float th = 3.841;
     const float thScore = 5.991;

     const float invSigmaSquare = 1.0/(sigma*sigma);

     //corres是vector3d 没有归一化
     for(int i=0; i< int(corres.size()); i++)
     {
         //bool bIn = true;

         //const cv::KeyPoint &kp1 = mvKeys1[mvMatches12[i].first];
         //const cv::KeyPoint &kp2 = mvKeys2[mvMatches12[i].second];

         //const float u1 = kp1.pt.x;
         //const float v1 = kp1.pt.y;
         //const float u2 = kp2.pt.x;
         //const float v2 = kp2.pt.y;

         //应该是用像素坐标，而不是归一化坐标
         Vector3d P1=(corres[i].first(0),corres[i].first(1),corres[i].first(2));
         Vector3d P2=(corres[i].second(0),corres[i].second(1),corres[i].second(2));
         Vector3d p1=K*P1;//相机坐标系转像素坐标系
         Vector3d p2=k*P2;

         const float u1 =p1.x;//像素坐标
         const float v1 =p1.y;
         const float u2 =p2.x;
         const float v2 =p2.y;

         // Reprojection error in second image
         // l2=F21x1=(a2,b2,c2)

         const float a2 = f11*u1+f12*v1+f13;
         const float b2 = f21*u1+f22*v1+f23;
         const float c2 = f31*u1+f32*v1+f33;

         const float num2 = a2*u2+b2*v2+c2;//理论上这个值应该为0，点(u2,v2,1)到极线a2*x+b2*y+c2=0的距离 不理解

         const float squareDist1 = num2*num2/(a2*a2+b2*b2);

         const float chiSquare1 = squareDist1*invSigmaSquare;

         if(chiSquare1>th) {
            // bIn = false;
             score = 0
         }
         else
             score += thScore - chiSquare1;

         // Reprojection error in first image
         // l1 =x2tF21=(a1,b1,c1)

         const float a1 = f11*u2+f21*v2+f31;
         const float b1 = f12*u2+f22*v2+f32;
         const float c1 = f13*u2+f23*v2+f33;

         const float num1 = a1*u1+b1*v1+c1;

         const float squareDist2 = num1*num1/(a1*a1+b1*b1);

         const float chiSquare2 = squareDist2*invSigmaSquare;

         if(chiSquare2>th)
         {
            // bIn = false;
             score=0;

         }

         else
             score += thScore - chiSquare2;

       //  if(bIn)
        //     vbMatchesInliers[i]=true;
        // else
         //    vbMatchesInliers[i]=false;
     }

     return score;

 }
float InitialEXRotation::CheckHomography(cv::Mat &H, float sigma)
{
    cv::Mat Hinv;
    Hinv=H.inv();

    const float h11 = H.at<float>(0,0);
    const float h12 = H.at<float>(0,1);
    const float h13 = H.at<float>(0,2);
    const float h21 = H.at<float>(1,0);
    const float h22 = H.at<float>(1,1);
    const float h23 = H.at<float>(1,2);
    const float h31 = H.at<float>(2,0);
    const float h32 = H.at<float>(2,1);
    const float h33 = H.at<float>(2,2);

    const float h11inv = Hinv.at<float>(0,0);
    const float h12inv = Hinv.at<float>(0,1);
    const float h13inv = Hinv.at<float>(0,2);
    const float h21inv = Hinv.at<float>(1,0);
    const float h22inv = Hinv.at<float>(1,1);
    const float h23inv = Hinv.at<float>(1,2);
    const float h31inv = Hinv.at<float>(2,0);
    const float h32inv = Hinv.at<float>(2,1);
    const float h33inv = Hinv.at<float>(2,2);



    float score = 0;

    const float th = 5.991;

    const float invSigmaSquare = 1.0/(sigma*sigma);

    for (int i = 0; i < int(corres.size()); i++)//匹配点对的数量
    {
       // bool bIn = true;

       // const cv::KeyPoint &kp1 = mvKeys1[mvMatches12[i].first];
        //const cv::KeyPoint &kp2 = mvKeys2[mvMatches12[i].second];

        //const float u1 = corres[i].first(0);
       // const float v1 = corres[i].first(1);
       // const float u2 = corres[i].second(0);
       // const float v2 = corres[i].second(1);
        //ll.push_back(cv::Point2f(corres[i].first(0), corres[i].first(1)));//取点的x,y坐标push进vector中
       // rr.push_back(cv::Point2f(corres[i].second(0), corres[i].second(1)));
        //应该是用像素坐标，而不是归一化坐标
        Vector3d P1=(corres[i].first(0),corres[i].first(1),corres[i].first(2));
        Vector3d P2=(corres[i].second(0),corres[i].second(1),corres[i].second(2));
        Vector3d p1=K*P1;//相机坐标系转像素坐标系
        Vector3d p2=k*P2;

        const float u1 =p1.x;//像素坐标
        const float v1 =p1.y;
        const float u2 =p2.x;
        const float v2 =p2.y;


        // Reprojection error in first image
        // x2in1 = H12*x2

        const float w2in1inv = 1.0/(h31inv*u2+h32inv*v2+h33inv);
        const float u2in1 = (h11inv*u2+h12inv*v2+h13inv)*w2in1inv;
        const float v2in1 = (h21inv*u2+h22inv*v2+h23inv)*w2in1inv;

        const float squareDist1 = (u1-u2in1)*(u1-u2in1)+(v1-v2in1)*(v1-v2in1);

        const float chiSquare1 = squareDist1*invSigmaSquare;

        if(chiSquare1>th)
            score=0;
        else
            score += th - chiSquare1;

        // Reprojection error in second image
        // x1in2 = H21*x1

        const float w1in2inv = 1.0/(h31*u1+h32*v1+h33);
        const float u1in2 = (h11*u1+h12*v1+h13)*w1in2inv;
        const float v1in2 = (h21*u1+h22*v1+h23)*w1in2inv;

        const float squareDist2 = (u2-u1in2)*(u2-u1in2)+(v2-v1in2)*(v2-v1in2);

        const float chiSquare2 = squareDist2*invSigmaSquare;

        if(chiSquare2>th)
            score=0;
        else
            score += th - chiSquare2;

       /* if(bIn)
            vbMatchesInliers[i]=true;
        else
            vbMatchesInliers[i]=false;
            */
    }

    return score;

}

void InitialEXRotation::decomposeH(cv::Mat& H)
{
    cv::Mat K=(Mat_<double>(3,3)<<461.6,0,363.0,0,460.3,248.1,0,0,1);
    cv::Mat invK = K.inv();
    cv::Mat A = invK*H*K;

    cv::Mat U,w,Vt,V;
    cv::SVD::compute(A,w,U,Vt,cv::SVD::FULL_UV);
    V=Vt.t();

    float s = cv::determinant(U)*cv::determinant(Vt);

    float d1 = w.at<float>(0);
    float d2 = w.at<float>(1);
    float d3 = w.at<float>(2);

    if(d1/d2<1.00001 || d2/d3<1.00001)
    {
        return false;
    }

    vector<cv::Mat> vR, vt, vn;
    vR.reserve(8);
    vt.reserve(8);
    vn.reserve(8);

    //n'=[x1 0 x3] 4 posibilities e1=e3=1, e1=1 e3=-1, e1=-1 e3=1, e1=e3=-1
    float aux1 = sqrt((d1*d1-d2*d2)/(d1*d1-d3*d3));
    float aux3 = sqrt((d2*d2-d3*d3)/(d1*d1-d3*d3));
    float x1[] = {aux1,aux1,-aux1,-aux1};
    float x3[] = {aux3,-aux3,aux3,-aux3};

    //case d'=d2
    float aux_stheta = sqrt((d1*d1-d2*d2)*(d2*d2-d3*d3))/((d1+d3)*d2);

    float ctheta = (d2*d2+d1*d3)/((d1+d3)*d2);
    float stheta[] = {aux_stheta, -aux_stheta, -aux_stheta, aux_stheta};

    for(int i=0; i<4; i++)
    {
        cv::Mat Rp=cv::Mat::eye(3,3,CV_32F);
        Rp.at<float>(0,0)=ctheta;
        Rp.at<float>(0,2)=-stheta[i];
        Rp.at<float>(2,0)=stheta[i];
        Rp.at<float>(2,2)=ctheta;

        cv::Mat R = s*U*Rp*Vt;
        vR.push_back(R);

        cv::Mat tp(3,1,CV_32F);
        tp.at<float>(0)=x1[i];
        tp.at<float>(1)=0;
        tp.at<float>(2)=-x3[i];
        tp*=d1-d3;

        cv::Mat t = U*tp;
        vt.push_back(t/cv::norm(t));

        cv::Mat np(3,1,CV_32F);
        np.at<float>(0)=x1[i];
        np.at<float>(1)=0;
        np.at<float>(2)=x3[i];

        cv::Mat n = V*np;
        if(n.at<float>(2)<0)
            n=-n;
        vn.push_back(n);
    }

    //case d'=-d2
    float aux_sphi = sqrt((d1*d1-d2*d2)*(d2*d2-d3*d3))/((d1-d3)*d2);

    float cphi = (d1*d3-d2*d2)/((d1-d3)*d2);
    float sphi[] = {aux_sphi, -aux_sphi, -aux_sphi, aux_sphi};

    for(int i=0; i<4; i++)
    {
        cv::Mat Rp=cv::Mat::eye(3,3,CV_32F);
        Rp.at<float>(0,0)=cphi;
        Rp.at<float>(0,2)=sphi[i];
        Rp.at<float>(1,1)=-1;
        Rp.at<float>(2,0)=sphi[i];
        Rp.at<float>(2,2)=-cphi;

        cv::Mat R = s*U*Rp*Vt;
        vR.push_back(R);

        cv::Mat tp(3,1,CV_32F);
        tp.at<float>(0)=x1[i];
        tp.at<float>(1)=0;
        tp.at<float>(2)=x3[i];
        tp*=d1+d3;

        cv::Mat t = U*tp;
        vt.push_back(t/cv::norm(t));

        cv::Mat np(3,1,CV_32F);
        np.at<float>(0)=x1[i];
        np.at<float>(1)=0;
        np.at<float>(2)=x3[i];

        cv::Mat n = V*np;
        if(n.at<float>(2)<0)
            n=-n;
        vn.push_back(n);
    }


}
int InitialEXRotation::CheckRT(const cv::Mat &R, const cv::Mat &t, const vector<cv::KeyPoint> &vKeys1, const vector<cv::KeyPoint> &vKeys2,
                         const vector<Match> &vMatches12, vector<bool> &vbMatchesInliers,
                         const cv::Mat &K, vector<cv::Point3f> &vP3D, float th2, vector<bool> &vbGood, float &parallax)
{
    // Calibration parameters
    const float fx = K.at<float>(0,0);
    const float fy = K.at<float>(1,1);
    const float cx = K.at<float>(0,2);
    const float cy = K.at<float>(1,2);

    vbGood = vector<bool>(corres.size(),false);//上面传递的传corres进来
    //vP3D.resize(vKeys1.size());

    //vector<float> vCosParallax;
   // vCosParallax.reserve(vKeys1.size());

    // Camera 1 Projection Matrix K[I|0]
    cv::Mat P1(3,4,CV_32F,cv::Scalar(0));
    K.copyTo(P1.rowRange(0,3).colRange(0,3));

    cv::Mat O1 = cv::Mat::zeros(3,1,CV_32F);

    // Camera 2 Projection Matrix K[R|t]
    cv::Mat P2(3,4,CV_32F);
    R.copyTo(P2.rowRange(0,3).colRange(0,3));
    t.copyTo(P2.rowRange(0,3).col(3));
    P2 = K*P2;

    cv::Mat O2 = -R.t()*t;

    int nGood=0;

    for(size_t i=0, iend=int(corres.size());i<iend;i++)
    {
       // if(!vbMatchesInliers[i])
        //    continue;

       // const cv::KeyPoint &kp1 = vKeys1[vMatches12[i].first];
       // const cv::KeyPoint &kp2 = vKeys2[vMatches12[i].second];
        cv::Mat p3dC1;
        //把keypoint换位corres的ll和rr

        Triangulate(kp1,kp2,P1,P2,p3dC1);
        //testTriangulation(ll, rr, R1, t1) 这两函数功能应该是相似的，统一一下，用一个吧

        if(!isfinite(p3dC1.at<float>(0)) || !isfinite(p3dC1.at<float>(1)) || !isfinite(p3dC1.at<float>(2)))
        {
            vbGood[vMatches12[i].first]=false;
            continue;
        }

        // Check parallax
        cv::Mat normal1 = p3dC1 - O1;
        float dist1 = cv::norm(normal1);

        cv::Mat normal2 = p3dC1 - O2;
        float dist2 = cv::norm(normal2);

        float cosParallax = normal1.dot(normal2)/(dist1*dist2);

        // Check depth in front of first camera (only if enough parallax, as "infinite" points can easily go to negative depth)
        if(p3dC1.at<float>(2)<=0 && cosParallax<0.99998)
            continue;

        // Check depth in front of second camera (only if enough parallax, as "infinite" points can easily go to negative depth)
        cv::Mat p3dC2 = R*p3dC1+t;

        if(p3dC2.at<float>(2)<=0 && cosParallax<0.99998)
            continue;

        // Check reprojection error in first image
        float im1x, im1y;
        float invZ1 = 1.0/p3dC1.at<float>(2);
        im1x = fx*p3dC1.at<float>(0)*invZ1+cx;
        im1y = fy*p3dC1.at<float>(1)*invZ1+cy;

        float squareError1 = (im1x-kp1.pt.x)*(im1x-kp1.pt.x)+(im1y-kp1.pt.y)*(im1y-kp1.pt.y);

        if(squareError1>th2)
            continue;

        // Check reprojection error in second image
        float im2x, im2y;
        float invZ2 = 1.0/p3dC2.at<float>(2);
        im2x = fx*p3dC2.at<float>(0)*invZ2+cx;
        im2y = fy*p3dC2.at<float>(1)*invZ2+cy;

        float squareError2 = (im2x-kp2.pt.x)*(im2x-kp2.pt.x)+(im2y-kp2.pt.y)*(im2y-kp2.pt.y);

        if(squareError2>th2)
            continue;

        vCosParallax.push_back(cosParallax);
        vP3D[vMatches12[i].first] = cv::Point3f(p3dC1.at<float>(0),p3dC1.at<float>(1),p3dC1.at<float>(2));
        nGood++;

        if(cosParallax<0.99998)
            vbGood[vMatches12[i].first]=true;
    }

    if(nGood>0)
    {
        sort(vCosParallax.begin(),vCosParallax.end());

        size_t idx = min(50,int(vCosParallax.size()-1));
        parallax = acos(vCosParallax[idx])*180/CV_PI;
    }
    else
        parallax=0;

    return nGood;
}