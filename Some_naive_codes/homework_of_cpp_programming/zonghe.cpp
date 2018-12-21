#include <iostream>
#include <iomanip>
#include "opencv2/opencv.hpp"

/*
[程序名称]
_zonghe_1.cpp

[程序风格]
CXX STANDARD 17

[开发平台]
Visual Studio 2017 Community

[其他说明]
首先，为了锻炼自己的能力，我没有使用老师给出的框架，而是从头写了一份。
在作业中我想尽量让程序c++一点，所以使用了一些c++的方法代替c，这确实会超出目前所
学范围，但本人会指出这种写法对应的比较传统的C写法，以证明我没有偷懒。以下是一些对
应关系：
1. array<array<int, MAXSIZE>, MAXSIZE> 对应 int[MAXSIZE][MAXSIZE]
2. paramType& a = b					   对应 paramType* a = &b
3. #define a b						   对应 constexpr Type a = b

*/

constexpr uint32_t MAXSIZE = 100;
// 在现代C++风格中，对于常量提倡使用constexpr关
// 键字进行描述,特此说明。
typedef std::array<std::array<int, MAXSIZE>, MAXSIZE> MAT;

using namespace std;


// DEFINE OF FUNCS
inline void menu_print();
void menu();

void mat_print(const MAT&, uint32_t, uint32_t);
void mat_build(MAT&, uint32_t, uint32_t);

void add_mat();
void num_multi_mat();
void trans();
void dot();
void hadamard_multi();
void conv2d();
void conv2d_apply();

void filter_apply(const cv::Mat&, cv::Mat&, const cv::Mat& filter, array<uint16_t, 2> = {1, 1});

// MAIN FUNCTION
int main() 
{
	menu();
	system("pause");
}

// IMPLEMENTATION OF FUNCS
inline void menu_print()
{
	cout << endl;
	cout << "*************************************************" << endl
		 << " *  1 矩阵加法     2 矩阵数乘       3 矩阵转置  * " << endl
		 << " *  4 矩阵乘法     5 Hadamard乘积   6 矩阵卷积  * " << endl
		 << " *  7 卷积应用     8 退出系统                   * " << endl
		 << "*************************************************" << endl
		 << "选择菜单项 <0 ~ 7>:" << endl;
}

void menu()
{
	uint16_t opt = 8;
	menu_print();
	while (cin >> opt && opt != 8) {
		switch (opt) {
		case 1:
			add_mat();
			break;
		case 2:
			num_multi_mat();
			break;
		case 3:
			trans();
			break;
		case 4:
			dot();
			break;
		case 5:
			hadamard_multi();
			break;
		case 6:
			conv2d();
			break;
		case 7:
			conv2d_apply();
			break;
		}
		menu_print();
	}
	cout << endl << "进程结束，谢谢使用！" << endl;
}

void mat_print(const MAT& mat_, uint32_t row, uint32_t col) 
{	
	cout << endl;
	for (size_t i = 0; i < row; i++)
	{	
		cout << "| ";
		for (size_t j = 0; j < col; j++)
		{
			cout << setw(5) << mat_[i][j] << ' ';
		}
		cout << "|" << endl;
	}
}

void mat_build(MAT& ret_mat, uint32_t row, uint32_t col) 
{
	for (size_t i = 0; i < row; i++)
	{
		for (size_t j = 0; j < col; j++)
		{	
			cout << "目前请输入第[" << i + 1 << "]行[" << j + 1 << "]列元素\t";
			cin >> ret_mat[i][j];
		}
	}
}

void add_mat()
{
	cout << "----------矩阵相加---------" << endl;
	auto mat1 = MAT();
	auto mat2 = MAT();
	uint32_t row, col;

	cout << "请输入两个矩阵的[行数]和[列数]" << endl;
	cin >> row >> col;

	cout << "\t请输入第一个矩阵的元素" << endl;
	mat_build(mat1, row, col);

	cout << "\t请输入第二个矩阵的元素" << endl;
	mat_build(mat2, row, col);

	cout << "相加后的结果" << endl;
	cout << endl;
	for (size_t i = 0; i < row; i++)
	{
		cout << "| ";
		for (size_t j = 0; j < col; j++)
		{
			cout << setw(5) << mat1[i][j]+mat2[i][j] << ' ';
		}
		cout << "|" << endl;
	}
}

void num_multi_mat()
{
	cout << "-----------数乘-----------" << endl;
	auto mat1 = MAT();
	uint32_t row, col;

	int num;
	cout << "请输入\"数乘\"中的\"数\"" << endl;
	cin >> num;

	cout << "请输入矩阵的[行数]和[列数]" << endl;
	cin >> row >> col;

	cout << "\t请输入第一个矩阵的元素" << endl;
	mat_build(mat1, row, col);

	cout << "数乘的结果" << endl;
	cout << endl;
	for (size_t i = 0; i < row; i++)
	{
		cout << "| ";
		for (size_t j = 0; j < col; j++)
		{
			cout << setw(5) << mat1[i][j]*num << ' ';
		}
		cout << "|" << endl;
	}
}

void trans()
{
	cout << "-----------转置-----------" << endl;
	auto mat1 = MAT();
	uint32_t row, col;

	cout << "请输入矩阵的[行数]和[列数]" << endl;
	cin >> row >> col;

	cout << "\t请输入第一个矩阵的元素" << endl;
	mat_build(mat1, row, col);

	cout << "转置的结果" << endl;
	cout << endl;
	for (size_t i = 0; i < col; i++)
	{
		cout << "| ";
		for (size_t j = 0; j < row; j++)
		{
			cout << setw(5) << mat1[j][i] << ' ';
		}
		cout << "|" << endl;
	}
}

void dot()
{

	cout << "---------矩阵相乘---------" << endl;
	auto mat1 = MAT();
	auto mat2 = MAT();
	uint32_t row1, col1, row2, col2;

	cout << "请输入第一个矩阵的[行数]和[列数]" << endl;
	cin >> row1 >> col1;

	cout << "\t请输入第一个矩阵的元素" << endl;
	mat_build(mat1, row1, col1);

	cout << "请输入第二个矩阵的[行数]和[列数]" << endl;
	while (!(cin >> row2 >> col2 && row2 == col1))
	{
		cout << "第一个矩阵的列数应该和第二个矩阵的行数一样，请重新输入！" << endl;
		cin.clear();
		cin.ignore();
	};
	cout << "\t请输入第二个矩阵的元素" << endl;
mat_build(mat2, row2, col2);

cout << "相乘后的结果" << endl;
cout << endl;
for (size_t i = 0; i < row1; i++)
{
	cout << "| ";
	for (size_t j = 0; j < col2; j++)
	{
		int ans = 0;
		for (size_t k = 0; k < col1; k++)
		{
			ans += mat1[i][k] * mat2[k][j];
		}
		cout << setw(5) << ans << ' ';
	}
	cout << "|" << endl;
}
}

void hadamard_multi()
{
	cout << "---------Hadamard--------" << endl;
	auto mat1 = MAT();
	auto mat2 = MAT();
	uint32_t row, col;

	cout << "请输入两个矩阵的[行数]和[列数]" << endl;
	cin >> row >> col;

	cout << "\t请输入第一个矩阵的元素" << endl;
	mat_build(mat1, row, col);

	cout << "\t请输入第二个矩阵的元素" << endl;
	mat_build(mat2, row, col);

	cout << "Hadamard乘积后的结果" << endl;
	cout << endl;
	for (size_t i = 0; i < row; i++)
	{
		cout << "| ";
		for (size_t j = 0; j < col; j++)
		{
			cout << setw(5) << mat1[i][j] * mat2[i][j] << ' ';
		}
		cout << "|" << endl;
	}
}

void conv2d()
{
	cout << "---------矩阵卷积--------" << endl;

	uint32_t filter_size[2];
	uint16_t stride[2];

	cout << "请依次输入：[filter的行数]， [filter的列数], [行的步长stride]， [列的步长stride]" << endl;
	while (!(
		cin >> filter_size[0] >> filter_size[1] >> stride[0] >> stride[1]
		&& filter_size[0] > 0 && filter_size[1] > 0 && stride[0] > 0 && stride[1] > 0
		))
	{
		cout << "以上输入都应大于0，请重新输入" << endl;
		cin.clear();
		cin.ignore();
	}

	auto filter = MAT();
	auto base_mat = MAT();

	cout << "请输入filter的元素" << endl;
	mat_build(filter, filter_size[0], filter_size[1]);

	cout << "请输入被卷积的矩阵的[行数]和[列数]" << endl;
	uint16_t row, col;
	cin >> row >> col;
	cout << "请输入被卷积的矩阵的元素" << endl;
	mat_build(base_mat, row, col);

	// 输出的size是(original_size+2*padding-filter_size)/stride+1
	for (size_t i = 0; i < row - filter_size[0]; i += stride[0])
	{
		cout << "| ";
		for (size_t j = 0; j < col - filter_size[1]; j += stride[1])
		{
			int ans = 0;
			for (size_t k = 0; k < filter_size[0]; k++)
			{
				for (size_t n = 0; n < filter_size[1]; n++)
				{
					ans += filter[k][n] * base_mat[i + k][j + n];
				}
			}
			cout << setw(5) << ans << ' ';
		}
		cout << "|" << endl;
	}
}

void filter_apply(const cv::Mat& input, cv::Mat& output, const cv::Mat& filter, array<uint16_t, 2> stride)
{
	uint16_t new_row = (input.rows - filter.rows) / stride[0] + 1;
	uint16_t new_col = (input.cols - filter.cols) / stride[1] + 1;
	output.create(new_row, new_col, input.type());
	for (size_t i = 0; i < new_row; i+=stride[0])
		for (size_t j = 0; j < new_col; j+=stride[1])
			for (size_t k = 0; k < output.channels(); k++)
			{// For each channel.
				int16_t pixel_val = 0;
				for (size_t m = 0; m < filter.rows; m++)
				{
					for (size_t n = 0; n < filter.cols; n++)
					{
						pixel_val += filter.at<char>(m, n)*static_cast<int16_t>(input.at<uchar>(i+m, output.channels()*j+k+n));
					}
				}
				output.at<uchar>(i, j*output.channels()+k) = cv::saturate_cast<uchar>(pixel_val);
			}
}

void conv2d_apply() {
	auto img = cv::imread("demolena.jpg", 0);
	cv::Mat img_[5];
	cv::Mat kernel[5];

	kernel[0] = (cv::Mat_<char>(3, 3) << 1, 1, 1, 1, -7, 1, 1, 1, 1);
	kernel[1] = (cv::Mat_<char>(3, 3) << -1, -1, -1, -1, 8, -1, -1, -1, -1);
	kernel[2] = (cv::Mat_<char>(3, 3) << -1, -1, 0, -1, 0, 1, 0, 1, 1);
	kernel[3] = (cv::Mat_<char>(5, 5) << 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
	kernel[4] = (cv::Mat_<char>(5, 5) << 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 4, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0);
	cv::imshow("Original", img);
	
	char window_names[5][16] = { "win1", "win2", "win3", "win4", "win5" };

	for (size_t i = 0; i < 5; i++)
	{
		// cv::filter2D(img, img_[i], img.depth(), kernel[i]); // OpenCV official way.
		filter_apply(img, img_[i], kernel[i]);				// Self-made approach.
		cv::imshow(window_names[i], img_[i]);
	}

	cv::waitKey();
	cv::destroyAllWindows();
}
