#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

Mat Match(Mat standard, Mat image) {
    Mat new_image(image.rows, image.cols, CV_8UC3);
    //将standard，image分别分成3个通道便于后续分通道处理
    Mat image_part[3], standard_part[3];
    split(image, image_part);
    split(standard, standard_part);
    for (int channel = 0; channel < 3; channel++) {
        Mat hist_image, hist_standard;
        //均衡化处理
        equalizeHist(image_part[channel], image_part[channel]);
        equalizeHist(standard_part[channel], standard_part[channel]);

        //获取两个均衡化图像的直方图
        int histsize = 256;
        float ranges[] = { 0,256 };
        const float* histRanges = { ranges };
        calcHist(&image_part[channel], 1, 0, Mat(), hist_image, 1, &histsize, &histRanges);
        calcHist(&standard_part[channel], 1, 0, Mat(), hist_standard, 1, &histsize, &histRanges);

        //计算两个均衡化图像直方图的累积概率
        float hist_image_sum[256] = { hist_image.at<float>(0) };
        float hist_standard_sum[256] = { hist_standard.at<float>(0) };
        for (int i = 1; i < 256; i++) {
            hist_image_sum[i] = hist_image_sum[i - 1] + hist_image.at<float>(i);
            hist_standard_sum[i] = hist_standard_sum[i - 1] + hist_standard.at<float>(i);
        }
        for (int i = 0; i < 256; i++) {
            hist_image_sum[i] = hist_image_sum[i] / (image_part[channel].rows * image_part[channel].cols);
            hist_standard_sum[i] = hist_standard_sum[i] / (standard_part[channel].rows * standard_part[channel].cols);
        }
        //两个累计概率之间的差值，用于找到最接近的点
        float abs_matrix[256][256];
        for (int i = 0; i < 256; i++) {
            for (int j = 0; j < 256; j++) {
                //获取绝对值
                abs_matrix[i][j] = fabs(hist_image_sum[i] - hist_standard_sum[j]);
            }
        }
        Mat lut(1, 256, CV_8U);
        for (int i = 0; i < 256; i++) {
            //查找源灰度级为i的映射灰度和i的累积概率差最小(灰度接近)的规定化灰度
            //i对应image，j对应standard
            float min = abs_matrix[i][0];
            int index = 0;
            for (int j = 0; j < 256; j++) {
                //找到这一行的对应的最小下标即为对应变换到的灰度值
                if (min > abs_matrix[i][j]) {
                    min = abs_matrix[i][j];
                    index = j;
                }
            }
            //将找到的对应灰度映射放到lut中便于后续映射
            lut.at<uchar>(i) = index;
        }
        //图像中进行映射
        Mat image_enhanced;
        LUT(image_part[channel], lut, image_enhanced);
        for (int i = 0; i < image.rows; i++) {
            for (int j = 0; j < image.cols; j++) {
                new_image.at<Vec3b>(i, j)[channel] = int(image_enhanced.at<uchar>(i, j));
            }
        }
    }
    return new_image;
}

int main()
{
    Mat standard = imread("standard.png", 1);
    Mat image = imread("image.png", 1);
    imshow("待匹配图像", image);
    waitKey();
    imshow("匹配的模板图像", standard);
    waitKey();
    Mat new_image = Match(standard, image);
    imshow("直方图匹配结果", new_image);
    waitKey();

    //Mat standard = imread("standard.png", 1);
    //Mat image0 = imread("new_mss_收.tif", 1);
    //Mat image = image0(Range(1000, 1300), Range(1000, 1300));
    //imshow("待匹配图像", image);
    //waitKey();
    //imshow("匹配的模板图像", standard);
    //waitKey();
    //Mat new_image = Match(standard, image);
    //imshow("直方图匹配结果", new_image);
    //waitKey();
    return 0;
}