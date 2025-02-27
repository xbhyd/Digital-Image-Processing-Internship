#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

void sort(vector<int>& num) {
	int temp = 0;
	for (int i = 0; i < num.size() - 1; i++) {
		for (int j = 0; j < num.size() - 1 - i; j++) {
			if (num[j] > num[j + 1]) {
				temp = num[j];
				num[j] = num[j + 1];
				num[j + 1] = temp;
			}
		}
	}
}

//中值滤波
Mat MediaFilter(Mat image, int size) {
	int pos = (size - 1) / 2;
	Mat Newsrc;
	copyMakeBorder(image, Newsrc, pos, pos, pos, pos, BORDER_REFLECT_101);
	Mat new_image = Mat::zeros(image.rows, image.cols, image.type());
	for (int i = pos; i < image.rows + pos; i++) {//遍历输入图像中的每个像素（不包括边界像素）
		uchar* pt1 = new_image.ptr(i - pos);//获取指向new_image矩阵当前行的指针（pt1）
		for (int j = pos; j < image.cols + pos; j++) {
			vector<int> pix;
			for (int r = i - pos; r <= i + pos; r++) {
				//获取指向窗口内当前行的指针（pt2）
				uchar* pt2 = Newsrc.ptr(r);
				for (int c = j - pos; c <= j + pos; c++) {
					pix.push_back(pt2[c]);
				}
			}
			sort(pix);
			pt1[j - pos] = pix[(size * size - 1) / 2];
		}
	}
	return new_image;
}

//高通滤波
Mat LaplacianFilter4(Mat image) {
	Mat new_image = image.clone();
	int la = 0;
	for (int i = 1; i < (image.rows - 1); i++) {
		for (int j = 1; j < (image.cols - 1); j++) {
			la = 4 * image.at<uchar>(i, j) - image.at<uchar>(i + 1, j) - image.at<uchar>(i - 1, j) - image.at<uchar>(i, j + 1) - image.at<uchar>(i, j - 1);
			//new_image.at<uchar>(i, j) = saturate_cast<uchar>(new_image.at<uchar>(i, j) + la);
			new_image.at<uchar>(i, j) = saturate_cast<uchar>(la);
		}
	}
	return new_image(Range(1, image.rows - 1), Range(1, image.cols - 1));
}

Mat LaplacianFilter8(Mat image) {
	Mat new_image = image.clone();
	int la = 0;
	for (int i = 1; i < (image.rows - 1); i++) {
		for (int j = 1; j < (image.cols - 1); j++) {
			la = 8 * image.at<uchar>(i, j) - image.at<uchar>(i + 1, j) - image.at<uchar>(i - 1, j) - image.at<uchar>(i, j + 1) - image.at<uchar>(i, j - 1)
				- image.at<uchar>(i - 1, j - 1) - image.at<uchar>(i + 1, j + 1) - image.at<uchar>(i - 1, j + 1) - image.at<uchar>(i + 1, j - 1);
			//new_image.at<uchar>(i - 1, j - 1) = saturate_cast<uchar>(new_image.at<uchar>(i, j) + la);
			new_image.at<uchar>(i, j) = saturate_cast<uchar>(la);
		}
	}
	return new_image(Range(1, image.rows - 1), Range(1, image.cols - 1));
}

//低通滤波
Mat AverageFilter9(Mat image) {
	Mat new_image = image.clone();
	float num = 0.0;
	for (int i = 1; i < (image.rows - 1); i++) {
		for (int j = 1; j < (image.cols - 1); j++) {
			num = (1.0 / 9.0) * float(image.at<uchar>(i, j) + image.at<uchar>(i + 1, j) + image.at<uchar>(i - 1, j) 
				+ image.at<uchar>(i, j + 1) + image.at<uchar>(i, j - 1)
				+ image.at<uchar>(i - 1, j - 1) + image.at<uchar>(i + 1, j + 1) + image.at<uchar>(i - 1, j + 1) 
				+ image.at<uchar>(i + 1, j - 1));
			new_image.at<uchar>(i, j) = saturate_cast<uchar>(round(num));
		}
	}
	return new_image(Range(1, image.rows - 1), Range(1, image.cols - 1));
}

Mat AverageFilter16(Mat image) {
	Mat new_image = image.clone();
	float num = 0.0;
	for (int i = 1; i < (image.rows - 1); i++) {
		for (int j = 1; j < (image.cols - 1); j++) {
			num = 1.0 / 16.0 * float(4.0 * image.at<uchar>(i, j) + 2.0 * image.at<uchar>(i + 1, j) + 2.0 * image.at<uchar>(i - 1, j) + 2.0 * image.at<uchar>(i, j + 1) + 2.0 * image.at<uchar>(i, j - 1)
				+ image.at<uchar>(i - 1, j - 1) + image.at<uchar>(i + 1, j + 1) + image.at<uchar>(i - 1, j + 1) + image.at<uchar>(i + 1, j - 1));
			new_image.at<uchar>(i, j) = saturate_cast<uchar>(round(num));
		}
	}
	return new_image(Range(1, image.rows - 1), Range(1, image.cols - 1));
}

int main() {
	Mat image = imread("cat.jpg", 0);
	Mat new_image;
	new_image = MediaFilter(image, 3);
	namedWindow("cat");
	imshow("cat", image);
	waitKey();
	namedWindow("media_result");
	imshow("media_result", new_image);
	waitKey();

	//Mat image, new_image1, new_image2;
	//image = imread("new_pan.jpg", 0);
	//imshow("image", image);
	//waitKey();
	//new_image1 = LaplacianFilter4(image);
	//imshow("image_output", new_image1);
	//waitKey();
	//new_image2 = LaplacianFilter8(image);
	//imshow("image_output2", new_image2);
	//waitKey();

	//Mat image, new_image1, new_image2;
	//image = imread("new_pan.jpg", 0);
	//imshow("image", image);
	//waitKey();
	//new_image1 = AverageFilter9(image);
	//imshow("image_output", new_image1);
	//waitKey();
	//new_image2 = AverageFilter16(image);
	//imshow("image_output2", new_image2);
	//waitKey();

	return 0;
}