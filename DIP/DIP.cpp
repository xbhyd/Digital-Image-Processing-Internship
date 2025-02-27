#include <iostream>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

Mat pre_process_pan()
{
	Mat pan = imread("pan.tif", IMREAD_UNCHANGED);
	pan.convertTo(pan, CV_8UC1, 1.0 / 4.0, 0);

	namedWindow("pan", WINDOW_NORMAL);
	imshow("pan", pan);
	waitKey();

	Mat subset = pan(Range(7800, 8000), Range(7800, 8000));
	//namedWindow("pan", WINDOW_NORMAL);
	//imwrite("image.jpg", subset);
	//imshow("subset", subset);
	//waitKey();
	return subset;
}

Mat pre_process_color()
{
	char imageName[] = "mss.tif";
	Mat mss = imread(imageName, IMREAD_UNCHANGED);

	if (mss.empty())
	{
		fprintf(stderr, "Can not load image %s\n", imageName);
		waitKey(6000);
	}

	Mat b_g_r_ir[4], color_arr[3], color;
	split(mss, b_g_r_ir);

	color_arr[0] = b_g_r_ir[0];
	color_arr[1] = b_g_r_ir[1];
	color_arr[2] = b_g_r_ir[2];
	merge(color_arr, 3, color);

	//各波段数值范围大概在0-1001之间，所以建议转至8比特时，除以4
	color.convertTo(color, CV_8UC3, 1.0 / 4.0, 0);

	//若仅需要裁剪部分做后续处理
	Mat subset = color(Range(1100, 1200), Range(1100, 1200));
	imshow("subset", subset);
	//subset共享color的部分数据，前面为行范围，后面为列范围
	//imwrite("mss_subset.bmp", subset);
	//16U的数据可以保存至png、jpeg2000或tiff文件，其中png和jpeg2000需要设置压缩参数前面为行范围，后面为列范围
	return subset;
}

unsigned long int SearchNum(Mat image,int gray_num,int channel) {
	int height = image.rows;
	int width = image.cols;
	unsigned long int sum = 0;
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			if (channel == 0) {
				if (image.at<uchar>(i, j) == gray_num) {
					sum += 1;
				}
				continue;
			}
			if (image.at<Vec3b>(i, j)[channel] == gray_num) {
				sum += 1;
			}
		}
	}
	return sum;
}

int GrayHistogramMin(Mat image,int channel) {
	unsigned long int min = 0;
	int height = image.rows;
	int width = image.cols;
	for (int j = 0; j < 256; j++) {
		min = min + SearchNum(image, j, channel);
		if (static_cast<double>(min) / height / width > 0.02) {
			return j;
		}
	}
}

int GrayHistogramMax(Mat image,int channel) {
	unsigned long int max = 0;
	int height = image.rows;
	int width = image.cols;
	for (int i = 255; i > 0; i--) {
		max = max + SearchNum(image, i, channel);
		if (static_cast<double>(max) / height / width > 0.02) {
			return i;
		}
	}
}

Mat PrintNewPhoto(Mat image) {
	int height = image.rows;
	int width = image.cols;
	int* gray_num = new int[image.channels() * 2];
	for (int i = 0; i < image.channels(); i++) {
		gray_num[2*i] = GrayHistogramMin(image, i);
	}
	for (int i = 0; i < image.channels(); i++) {
		gray_num[2 * i + 1] = GrayHistogramMax(image, i);
	}
	Mat new_image = image.clone();
	if (image.channels() == 1) {
		double k = 255 / (static_cast<double>(gray_num[1]) - gray_num[0]);
		for (int m = 0; m < height; m++) {
			for (int n = 0; n < width; n++) {
				if (image.at<uchar>(m, n) < gray_num[0]) {
					new_image.at<uchar>(m, n) = 0;
				}
				else if (image.at<uchar>(m, n) > gray_num[1])
				{
					new_image.at<uchar>(m, n) = 255;
				}
				else {
					new_image.at<uchar>(m, n) = round((int(image.at<uchar>(m, n)) - gray_num[0]) * k);
				}
			}
		}
	}
	else {
		for (int i = 0; i < image.channels(); i++) {
			double k = 255 / (static_cast<double>(gray_num[2 * i + 1]) - gray_num[2 * i]);
			cout << "k=" << k << endl;
			for (int m = 0; m < height; m++) {
				for (int n = 0; n < width; n++) {
					cout << "m=" << m << " n=" << n << " i=" << i << endl;
					if (image.at<Vec3b>(m, n)[i] < gray_num[2 * i]) {
						cout << int(image.at<Vec3b>(m, n)[i]) << endl;
						new_image.at<Vec3b>(m, n)[i] = 0;
					}
					else if (image.at<Vec3b>(m, n)[i] > gray_num[2 * i + 1])
					{
						cout << int(image.at<Vec3b>(m, n)[i]) << endl;
						new_image.at<Vec3b>(m, n)[i] = 255;
					}
					else {
						cout << int(image.at<Vec3b>(m, n)[i]) << endl;
						new_image.at<Vec3b>(m, n)[i] = round((int(image.at<Vec3b>(m, n)[i]) - gray_num[2 * i]) * k);
					}
				}
			}
		}
	}
	delete gray_num;
	return new_image;
}

int main()
{
	Mat photo = pre_process_pan();
	imshow("picture", photo);
	waitKey();
	Mat new_photo = PrintNewPhoto(photo);
	namedWindow("new_image", WINDOW_NORMAL);
	imshow("new_image", new_photo);
	waitKey();
	return 0;
}