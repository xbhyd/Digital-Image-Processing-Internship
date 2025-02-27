// 第1种方法
//#include <iostream>
//#include<stdlib.h>
//#include"opencv2/opencv.hpp"
//#include<opencv2/core.hpp>
//#include<opencv2/highgui/highgui.hpp>
//
//using namespace std;
//using namespace cv;
//
////角二阶矩
//double Get_SecMoment(int length, double(*p)[16]) {//传入灰度共生矩阵、矩阵边长
//    double SecMoment = 0.0;//定义角二阶矩
//    for (int i = 0; i < length; i++) {
//        for (int j = 0; j < length; j++) {
//            SecMoment += p[i][j] * p[i][j];
//        }
//    }
//    return SecMoment;
//}
////惯性矩
//double Get_InaMoment(int length, double(*p)[16])
//{
//    //传入灰度共生矩阵、矩阵边长
//    double InaMoment = 0.0;//定义惯性矩
//    for (int i = 0; i < length; i++)
//    {
//        for (int j = 0; j < length; j++)
//        {
//            InaMoment += (double)(j - i) * (j - i) * p[i][j];
//        }
//    }
//    return InaMoment;
//}
//
////逆差矩
//double Get_inverse(int length, double(*p)[16])
//{
//    double inverse = 0;
//    for (int i = 0; i < length; i++)
//        for (int j = 0; j < length; j++)
//        {
//            inverse = p[i][j] / (1 + (double)(i - j) * (i - j)) + inverse;
//        }
//    return inverse;
//}
//
////相关
//double Get_relativity(int length, double(*p)[16]) {
//    double u1 = 0, u2 = 0, delta1 = 0, delta2 = 0;
//    double s1 = 0, temp = 0;
//    for (int i = 0; i < length; i++) {
//        temp = 0;
//        for (int j = 0; j < length; j++)
//        {
//            temp += p[i][j];
//        }
//        u1 += temp * i;
//    }
//    temp = 0;
//    for (int j = 0; j < length; j++) {
//        temp = 0;
//        for (int i = 0; i < length; i++)
//        {
//            temp = temp + p[i][j];
//        }
//        u2 += (temp * j);
//    }
//    temp = 0;
//    for (int i = 0; i < length; i++) {
//        temp = 0;
//        for (int j = 0; j < length; j++)
//        {
//            temp = temp + p[i][j];
//        }
//        delta1 += (i - u1) * (i - u1) * temp;
//    }
//    temp = 0;
//    for (int j = 0; j < length; j++)
//    {
//        temp = 0;
//        for (int i = 0; i < length; i++)
//        {
//            temp = temp + p[i][j];
//        }
//        delta2 += (j - u2) * (j - u2) * temp;
//    }
//    temp = 0;
//    for (int i = 0; i < length; i++)
//    {
//        for (int j = 0; j < length; j++)
//        {
//            temp += i * j * p[i][j];
//        }
//    }
//    double relativity = (temp - u1 * u2) / delta1 / delta2;
//    return relativity;
//}
//
////熵
//double entropy(int length, double(*p)[16])
//{
//    double entropy = 0;
//    for (int i = 0; i < length; i++)
//    {
//        for (int j = 0; j < length; j++)
//        {
//            if (p[i][j] <= 0) continue;
//            entropy = entropy - p[i][j] * (log(p[i][j]) / log(2));
//        }
//    }
//    return entropy;
//}
//
////求灰度共生矩阵及特征量
//void GreyMat(Mat my_image, double& feature, int choice)
//{
//    unsigned char* ptr = my_image.data;
//    int height = my_image.rows;
//    int width = my_image.cols;
//
//    //方向为0、45、90、135的四个共生矩阵
//    double p0[16][16] = { 0 };
//    double p45[16][16] = { 0 };
//    double p90[16][16] = { 0 };
//    double p135[16][16] = { 0 };
//
//    //计算图像最大灰度级
//    int pixel = 0;
//    double nMaxPixel = 0;
//    for (int i = 0; i < height; i++)
//    {
//        for (int j = 0; j < width; j++)
//        {
//            if (ptr[i * width + j] > nMaxPixel)
//            {
//                nMaxPixel = ptr[i * width + j];
//            }
//        }
//    }
//
//    //最大灰度级比最大灰度值大1
//    nMaxPixel = nMaxPixel + 1;
//
//    //将灰度级压缩到16级
//    for (int i = 0; i < height; i++)
//    {
//        for (int j = 0; j < width; j++)
//        {
//            pixel = ptr[i * width + j];
//            pixel = int(pixel * 16 / nMaxPixel);
//
//            ptr[i * width + j] = pixel;
//        }
//    }
//
//    //求各个方向的灰度共生矩阵
//    int pixel_0 = 0;    //暂时存放灰度值
//    int pixel_45 = 0;
//    int pixel_90 = 0;
//    int pixel_135 = 0;
//    int k = 0;
//    for (int i = 0; i < height; i++)
//    {
//        for (int j = 0; j < width; j++)
//        {
//            pixel = int(ptr[i * width + j]);
//
//            //防止越界
//            if (j < width - 1)
//            {
//                pixel_0 = ptr[i * width + j + 1];
//                p0[pixel][pixel_0]++;
//                p0[pixel_0][pixel]++;
//            }
//            if (j < width - 1 && i < height - 1)
//            {
//                pixel_45 = ptr[(i + 1) * width + j + 1];
//
//                p45[pixel][pixel_45]++;
//                p45[pixel_45][pixel]++;
//            }
//            if (i < height - 1)
//            {
//                pixel_90 = ptr[(i + 1) * width + j];
//
//                p90[pixel][pixel_90]++;
//                p90[pixel_90][pixel]++;
//            }
//            if (j > 0 && i < height - 1)
//            {
//                pixel_135 = ptr[(i + 1) * width + j];
//
//                p135[pixel][pixel_135]++;
//                p135[pixel_135][pixel]++;
//            }
//        }
//    }
//
//    //正规化处理
//    int sum0 = 2 * height * (width - 1);
//    int sum45 = 2 * (height - 1) * (width - 1);
//    int sum90 = 2 * (height - 1) * width;
//    int sum135 = 2 * (height - 1) * (width - 1);
//
//    for (int i = 0; i < 16; i++)
//    {
//        for (int j = 0; j < 16; j++)
//        {
//            p0[i][j] = p0[i][j] / sum0;
//            p45[i][j] = p45[i][j] / sum45;
//            p90[i][j] = p90[i][j] / sum90;
//            p135[i][j] = p135[i][j] / sum135;
//        }
//    }
//
//    double I1;
//    double I2;
//    double I3;
//    double I4;
//    if (choice == 1)
//    {
//        //角二阶矩
//        I1 = Get_SecMoment(16, p0);
//        I2 = Get_SecMoment(16, p45);
//        I3 = Get_SecMoment(16, p90);
//        I4 = Get_SecMoment(16, p135);
//    }
//    else if (choice == 2)
//    {
//        //惯性矩
//        I1 = Get_InaMoment(16, p0);
//        I2 = Get_InaMoment(16, p45);
//        I3 = Get_InaMoment(16, p90);
//        I4 = Get_InaMoment(16, p135);
//    }
//    else if (choice == 3)
//    {
//        //逆差矩
//        I1 = Get_inverse(16, p0);
//        I2 = Get_inverse(16, p45);
//        I3 = Get_inverse(16, p90);
//        I4 = Get_inverse(16, p135);
//    }
//    else if (choice == 4)
//    {
//        //相关
//        I1 = Get_relativity(16, p0);
//        I2 = Get_relativity(16, p45);
//        I3 = Get_relativity(16, p90);
//        I4 = Get_relativity(16, p135);
//    }
//    else if (choice == 5)
//    {
//        // 熵
//        I1 = entropy(16, p0);
//        I2 = entropy(16, p45);
//        I3 = entropy(16, p90);
//        I4 = entropy(16, p135);
//    }
//    feature = (I1 + I2 + I3 + I4) / 4.0;
//    //cout << "特征值为：" << feature << endl;
//}
//
//Mat GetFeature(Mat img, int choice, int pos)
//{
//    int height = img.rows;
//    int width = img.cols;
//    Mat new_img(img.rows - 2 * pos, img.cols - 2 * pos, CV_64FC1, Scalar(0.0));
//    Mat tmp_img = (Mat_<double>(2 * pos + 1, 2 * pos + 1));
//    for (int i = pos; i < height - pos; i++) {
//        for (int j = pos; j < width - pos; j++) {
//            for (int m = 0; m < tmp_img.rows; m++) {
//                for (int n = 0; n < tmp_img.cols; n++) {
//                    tmp_img.at<double>(m, n) = img.at<uchar>(i - pos + m, j - pos + n);
//                }
//            }
//            double feature = 0.0;
//            GreyMat(tmp_img, feature, choice);
//            new_img.at<double>(i - pos, j - pos) = feature;
//        }
//    }
//    return new_img;
//}
//
//Mat Match(Mat m_image, int pos) {
//    Mat p[5];
//    double average = 0.0;
//    double standard_minus = 0.0;
//    Mat fin(m_image.rows - 2 * pos, m_image.cols - 2 * pos, CV_8UC1, Scalar(0));
//    int sum = 0;
//    for (int i = 0; i < 5; i++) {
//        p[i] = GetFeature(m_image, i + 1, pos);
//
//        //下面是用来计算5个特征值的平均数、标准差的
//        for (int m = 0; m < p[i].rows; m++) {
//            for (int n = 0; n < p[i].cols; n++) {
//                average += p[i].at<double>(m, n);
//            }
//        }
//        average = average / p[i].rows / p[i].cols;
//        for (int m = 0; m < p[i].rows; m++) {
//            for (int n = 0; n < p[i].cols; n++) {
//                standard_minus += (average - p[i].at<double>(m, n)) * (average - p[i].at<double>(m, n));
//            }
//        }
//        standard_minus = sqrt(standard_minus / p[i].rows / p[i].cols);
//        cout << average << " " << standard_minus << endl;
//        average = 0.0;
//        standard_minus = 0.0;
//
//    }
//    for (int i = 0; i < fin.rows; i++)
//    {
//        for (int j = 0; j < fin.cols; j++)
//        {
//            //这里的阈值都由平均数加减标准差得到
//            if (p[0].at<double>(i, j) >= 0.2081962 && p[0].at<double>(i, j) <= 0.2773078) {
//                sum++;
//            }
//            if (p[1].at<double>(i, j) >= 20.0572 && p[1].at<double>(i, j) <= 41.3266) {
//                sum++;
//            }
//            if (p[2].at<double>(i, j) >= 0.4175424 && p[2].at<double>(i, j) <= 0.5004156) {
//                sum++;
//            }
//            if (p[3].at<double>(i, j) <= -0.00790553 && p[3].at<double>(i, j) >= -0.01379007) {
//                sum++;
//            }
//            if (p[4].at<double>(i, j) <= 2.907726 && p[4].at<double>(i, j) >= 2.471334) {
//                sum++;
//            }
//            if (sum > 4) {
//                fin.at<uchar>(i, j) = 255;
//            }
//            sum = 0;
//        }
//    }
//    return fin;
//}
//
//bool Judgement(Mat standard, Mat image) {//这个函数用来判断是否是林地
//    double probility1 = 0.0;
//    double probility2 = 0.0;
//    for (int m = 0; m < standard.rows; m++) {
//        for (int n = 0; n < standard.cols; n++) {
//            if (standard.at<uchar>(m, n) == 255) {
//                probility1++;
//            }
//        }
//    }
//    probility1 = probility1 / standard.rows / standard.cols;
//    //cout << probility1 << endl;
//    for (int m = 0; m < image.rows; m++) {
//        for (int n = 0; n < image.cols; n++) {
//            if (image.at<uchar>(m, n) == 255) {
//                probility2++;
//            }
//        }
//    }
//    probility2 = probility2 / image.rows / image.cols;
//    cout << probility2 << endl;
//    if (probility1 - probility2 < 0.03) {
//        return true;
//    }
//    else {
//        return false;
//    }
//}
//
//int main()
//{
//    Mat my_image = imread("pan.tif", IMREAD_GRAYSCALE);
//    //这个是取的标准林地
//    //Mat standard = my_image(Range(4350, 4600), Range(5050, 5300));
//    //Mat m_image = my_image(Range(3500, 3700), Range(3000, 3200));
//    Mat m_image = my_image(Range(2450, 2700), Range(9000, 9250));
//    cv::imshow("m_image", m_image);
//    cv::waitKey();
//    Mat fin = Match(m_image, 6);
//    cv::imshow("fin", fin);
//    cv::waitKey();
//    Mat standard = imread("standard.tif", 0);
//    bool answer = Judgement(standard, fin);
//    if (answer == true) {
//        cout << "是林地" << endl;
//    }
//    else {
//        cout << "不是林地" << endl;
//    }
//    return 0;
//}

// 第二种方法
#include <iostream>
#include<stdlib.h>
#include"opencv2/opencv.hpp"
#include<opencv2/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<Windows.h>
using namespace std;
using namespace cv;

//角二阶矩
double Get_SecMoment(int length, double(*p)[16])
{//传入灰度共生矩阵、矩阵边长
    double SecMoment = 0.0;//定义角二阶矩
    for (int i = 0; i < length; i++) {
        for (int j = 0; j < length; j++) {
            SecMoment += p[i][j] * p[i][j];
        }
    }
    return SecMoment;
}
//惯性矩
double Get_InaMoment(int length, double(*p)[16])
{
    //传入灰度共生矩阵、矩阵边长
    double InaMoment = 0.0;//定义惯性矩
    for (int i = 0; i < length; i++)
    {
        for (int j = 0; j < length; j++)
        {
            InaMoment += (double)(j - i) * (j - i) * p[i][j];
        }
    }
    return InaMoment;
}
//逆差矩
double Get_inverse(int length, double(*p)[16])
{
    double inverse = 0;
    for (int i = 0; i < length; i++)
        for (int j = 0; j < length; j++)
        {
            inverse = p[i][j] / (1 + (double)(i - j) * (i - j)) + inverse;
        }
    return inverse;
}

//相关
double Get_relativity(int length, double(*p)[16]) {
    double u1 = 0, u2 = 0, delta1 = 0, delta2 = 0;
    double s1 = 0, temp = 0;
    for (int i = 0; i < length; i++) {
        temp = 0;
        for (int j = 0; j < length; j++)
        {
            temp += p[i][j];
        }
        u1 += temp * i;
    }
    temp = 0;
    for (int j = 0; j < length; j++) {
        temp = 0;
        for (int i = 0; i < length; i++)
        {
            temp = temp + p[i][j];
        }
        u2 += (temp * j);
    }
    temp = 0;
    for (int i = 0; i < length; i++) {
        temp = 0;
        for (int j = 0; j < length; j++)
        {
            temp = temp + p[i][j];
        }
        delta1 += (i - u1) * (i - u1) * temp;
    }
    temp = 0;
    for (int j = 0; j < length; j++)
    {
        temp = 0;
        for (int i = 0; i < length; i++)
        {
            temp = temp + p[i][j];
        }
        delta2 += (j - u2) * (j - u2) * temp;
    }
    temp = 0;
    for (int i = 0; i < length; i++)
    {
        for (int j = 0; j < length; j++)
        {
            temp += i * j * p[i][j];
        }
    }
    double relativity = (temp - u1 * u2) / delta1 / delta2;
    return relativity;
}
//熵
double entropy(int length, double(*p)[16])
{
    double entropy = 0;
    for (int i = 0; i < length; i++)
    {
        for (int j = 0; j < length; j++)
        {
            if (p[i][j] <= 0) continue;
            entropy = entropy - p[i][j] * (log(p[i][j]) / log(2));
        }
    }
    return entropy;
}

//求灰度共生矩阵及特征量
void GreyMat(Mat my_image, double& feature, int choice)
{
    unsigned char* ptr = my_image.data;
    int height = my_image.rows;
    int width = my_image.cols;

    double p0[16][16] = { 0 };          //方向为0、45、90、135的四个共生矩阵
    double p45[16][16] = { 0 };
    double p90[16][16] = { 0 };
    double p135[16][16] = { 0 };


    //计算图像最大灰度级
    int pixel = 0;
    double nMaxPixel = 0;
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            if (ptr[i * width + j] > nMaxPixel)
            {
                nMaxPixel = ptr[i * width + j];
            }
        }
    }
    //最大灰度级比最大灰度值大1
    nMaxPixel = nMaxPixel + 1;

    //将灰度级压缩到16级
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            pixel = ptr[i * width + j];
            pixel = int(pixel * 16 / nMaxPixel);

            ptr[i * width + j] = pixel;
        }
    }

    //求各个方向的灰度共生矩阵
    int pixel_0 = 0;    //暂时存放灰度值
    int pixel_45 = 0;
    int pixel_90 = 0;
    int pixel_135 = 0;
    int k = 0;
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            pixel = int(ptr[i * width + j]);

            //防止越界
            if (j < width - 1)
            {
                pixel_0 = ptr[i * width + j + 1];
                p0[pixel][pixel_0]++;
                p0[pixel_0][pixel]++;
            }
            if (j < width - 1 && i < height - 1)
            {
                pixel_45 = ptr[(i + 1) * width + j + 1];

                p45[pixel][pixel_45]++;
                p45[pixel_45][pixel]++;
            }
            if (i < height - 1)
            {
                pixel_90 = ptr[(i + 1) * width + j];

                p90[pixel][pixel_90]++;
                p90[pixel_90][pixel]++;
            }
            if (j > 0 && i < height - 1)
            {
                pixel_135 = ptr[(i + 1) * width + j];

                p135[pixel][pixel_135]++;
                p135[pixel_135][pixel]++;
            }
        }
    }

    //正规化处理
    int sum0 = 2 * height * (width - 1);
    int sum45 = 2 * (height - 1) * (width - 1);
    int sum90 = 2 * (height - 1) * width;
    int sum135 = 2 * (height - 1) * (width - 1);

    for (int i = 0; i < 16; i++)
    {
        for (int j = 0; j < 16; j++)
        {
            p0[i][j] = p0[i][j] / sum0;
            p45[i][j] = p45[i][j] / sum45;
            p90[i][j] = p90[i][j] / sum90;
            p135[i][j] = p135[i][j] / sum135;
        }
    }

    double I1 = 0.0, I2 = 0.0, I3 = 0.0, I4 = 0.0;

    if (choice == 1)
    {
        //角二阶矩
        I1 = Get_SecMoment(16, p0);
        I2 = Get_SecMoment(16, p45);
        I3 = Get_SecMoment(16, p90);
        I4 = Get_SecMoment(16, p135);
    }
    else if (choice == 2)
    {
        //惯性矩
        I1 = Get_InaMoment(16, p0);
        I2 = Get_InaMoment(16, p45);
        I3 = Get_InaMoment(16, p90);
        I4 = Get_InaMoment(16, p135);
    }
    else if (choice == 3)
    {
        //逆差矩
        I1 = Get_inverse(16, p0);
        I2 = Get_inverse(16, p45);
        I3 = Get_inverse(16, p90);
        I4 = Get_inverse(16, p135);
    }
    else if (choice == 4)
    {
        I1 = Get_relativity(16, p0);
        I2 = Get_relativity(16, p45);
        I3 = Get_relativity(16, p90);
        I4 = Get_relativity(16, p135);
    }
    else if (choice == 5)
    {
        // 熵
        I1 = entropy(16, p0);
        I2 = entropy(16, p45);
        I3 = entropy(16, p90);
        I4 = entropy(16, p135);
    }
    feature = (I1 + I2 + I3 + I4) / 4.0;
    //  cout << "特征值"<<choice<<"为：" << feature << endl;
}

void GetFeature(Mat img, int pos, double feature_aver[5])//模板在图中移动,分割,最终返回示例图平均特征向量
{
    int height = img.rows;
    int width = img.cols;
    //Mat new_img(img.rows - 2 * pos, img.cols - 2 * pos, CV_64FC1, Scalar(0.0));
    Mat tmp_img = Mat::zeros(2 * pos + 1, 2 * pos + 1, CV_64FC1);
    double feature_sum[5] = { 0.0 };
    int num_x = 0, num_y = 0;
    for (int i = pos; i < height - pos; i = i + 2 * pos + 1) //中心位置移动
    {
        for (int j = pos; j < width - pos; j = j + 2 * pos + 1)
        {
            for (int m = 0; m < tmp_img.rows; m++)
            {
                for (int n = 0; n < tmp_img.cols; n++)
                {
                    tmp_img.at<double>(m, n) = img.at<uchar>(i - pos + m, j - pos + n);
                }
            }
            double feature[5] = { 0.0 };
            for (int t = 0; t < 5; t++)
            {
                double p = 0.0;
                GreyMat(tmp_img, p, t + 1);//返回单位图像灰度共生矩阵特征值
                feature[t] = p;
                feature_sum[t] += feature[t];
            }
            num_y++;
        }
        num_x++;
    }
    for (int i = 0; i < 5; i++)
    {
        feature_aver[i] = feature_sum[i] / (num_x * num_y);
    }

}

double dotProduct(const double* vec1, const double* vec2, int size) {
    double result = 0.0;
    for (int i = 0; i < size; i++) {
        result += vec1[i] * vec2[i];
    }
    return result;
}//向量点乘

double calculateNorm(const double* vec, int size) {
    double result = 0.0;
    for (int i = 0; i < size; i++) {
        result += vec[i] * vec[i];
    }
    return std::sqrt(result);
}//范数计算

double calculateCosineSimilarity(const double* vec1, const double* vec2, int size)
{
    double dot = dotProduct(vec1, vec2, size);
    double norm1 = calculateNorm(vec1, size);
    double norm2 = calculateNorm(vec2, size);

    return dot / (norm1 * norm2);
}//计算余弦相似度

Mat GetForest(Mat img, int pos, double feature_aver[5])
{
    int height = img.rows;
    int width = img.cols;
    Mat newimg = img.clone();
    Mat tmp_img = Mat::zeros(2 * pos + 1, 2 * pos + 1, CV_64FC1);
    int num_x = 0, num_y = 0;
    for (int i = pos; i < height - pos; i = i + 1) //中心位置移动
    {
        for (int j = pos; j < width - pos; j = j + 1)
        {
            for (int m = 0; m < tmp_img.rows; m++)
            {
                for (int n = 0; n < tmp_img.cols; n++)
                {
                    tmp_img.at<double>(m, n) = img.at<uchar>(i - pos + m, j - pos + n);
                }
            }
            double feature[5] = { 0.0 };
            for (int t = 0; t < 5; t++)
            {
                double p = 0.0;
                GreyMat(tmp_img, p, t + 1);//返回单位图像灰度共生矩阵特征值
                feature[t] = p;
            }
            double com = calculateCosineSimilarity(feature, feature_aver, 5);
            cout << com << endl;
            if (com < 0.9985)
            {
                for (int a = 0; a < tmp_img.rows; a++)
                {
                    for (int b = 0; b < tmp_img.cols; b++)
                        newimg.at<uchar>(i - pos + a, j - pos + b) = 0;
                }
            }
            else
            {
                for (int a = 0; a < tmp_img.rows; a++)
                {
                    for (int b = 0; b < tmp_img.cols; b++)
                        newimg.at<uchar>(i - pos + a, j - pos + b) = 255;
                }
            }//化为二值图像
        }
    }
    namedWindow("final", WINDOW_NORMAL);
    imshow("final", newimg);
    cv::waitKey();
    return newimg;
}

int main()
{
    Mat my_image = imread("pan1.bmp", IMREAD_GRAYSCALE);   // 读入图片 
    imshow("1", my_image);
    waitKey();
    double feature_ex[5] = { 0 };
    GetFeature(my_image, 8, feature_ex);
    cout << "特征向量为" << endl;
    for (int i = 0; i < 5; i++)cout << feature_ex[i] << " ";
    Mat r_image = imread("pan3.bmp", IMREAD_GRAYSCALE);   // 读入图片 
    imshow("2", r_image);
    waitKey();
    Mat newimg;
    newimg = GetForest(r_image, 8, feature_ex);
    /* cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(17, 17));
      cv::Mat erodedImage;
      cv::erode(newimg, erodedImage, element);//腐蚀
      cv::Mat dilatedImage;
      cv::dilate(erodedImage, dilatedImage, element);//膨胀
      namedWindow("Dilated Image", WINDOW_NORMAL);
      imshow("Dilated Image", dilatedImage);*/
    waitKey();
    return 0;
}