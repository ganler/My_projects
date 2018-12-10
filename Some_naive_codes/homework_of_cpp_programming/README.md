

# 高程大作业报告

![img](https://s1.ax1x.com/2018/12/10/FJJNsH.png)









#### 班级/专业： 电子二班/计算机科学与技术

#### 学号：	       1753070

#### 姓名：           刘佳伟

> 该报告为本人本科上高程的时候的一个report，感觉当时写的还是挺用心的。就是太中二了。。。。

## 设计思路与功能描述

#### [设计思路]

##### 程序的书写思路（以代码可读性、代码效率和代码健壮性为驱动）

- 关于可读性：

> **为什么要注重可读性？**
>
> > 如果代码还有被别人浏览的必要，那么该代码就应该注重可读性。代码是给人看的，如何书写规范清晰的代码不仅能让别人看得懂自己的代码，更能让自己在维护、检查代码的时候事半功倍。考虑到阅卷老师的不易，所以我在书写代码的时候十分注意代码的规范性。
>
> 在书写代码时，我注意了一下几点（主要参考了Google的开源项目代码风格指南）：
>
> - 代码布局：比如函数先声明再实现，不同类型的代码要放在一起。
> - 格式书写的统一：变量名统一用**kebab-case**命名法，大括号统一对齐等。
> - 程序语言的统一：所有代码均使用modern C++风格书写，尽量避免了使用C的风格，C++和C风格的代码混在一起会显得有点不够美观，而且确实C++中提供了许多C方法的替代品，并且更加安全和高效。某个大佬程序员曾说过，程序员还是有点洁癖为好。

使用IDE对整体代码进行归纳整理后可见整体代码结构非常清晰。不同类别的代码和不同类别的代码放在一起，加上合理的空格显得井井有条。

![img](https://s1.ax1x.com/2018/12/10/FJJULd.png)

- 关于代码效率：
  - 对于经常调用的函数，将其设置为内联函数。
  - 对于在编译阶段就可以获得的结果，使用`constexpr`关键字将其转化为编译阶段就能计算出结果的常量表达式。
  - 对于大型数据（如矩阵等内存大于8字节的），使用引用或指针的方法来提高代码的运行速度，同时也减小了内存的使用。而对于8字节（这里只讨论64位系统，64位系统下的指针是8字节的）以下的数据，直接使用拷贝传参的方法（因为指针的大小就是8字节，而解引用又会存在额外的开销）。By the way，`opencv`中的`cv::Mat`类型的数据是默认使用引用的方法在变量中传递的，因为`cv::Mat`存储的往往都是大型的动态数组。
  - 使用静态数据结构进行矩阵存储，提高运行效率。（静态数据结构对应的物理存储是连续的，这样存储的数据，CPU处理起来效率会比较高，类似于流水线）。
  - 同样，使用位运算可以大大提高程序的运行效率（即使是在开了很高级的优化的情况下），在图像处理中，减色函数(`ColorReduce`)的设计就可以使用这种方法，而由于本题我们使用的场景是用卷积对图像进行卷积，而将其写成位运算的代码并不方便，就算写出来可读性也会受损，所以依旧使用的是传统的方法。（减色函数一般都是对像素范围进行$2^n$的压缩，即让像素值整除$2^n$，使用位运算的方法即让数据对应的二进制右移n位即可）。
- 关于代码的健壮性：
  - 这里对输入环节进行了检查，对于不符合要求的输入，要求用户重新输入并且给出提示。使用输入检查的地方有：
    - 2矩阵相乘的合法性（第一个矩阵的列数应该等于第二个矩阵的行数）；
    - 矩阵相加和Hadamard乘积时，矩阵的size应该一样等。



##### 程序的逻辑思路

> - 使用`menu`函数控制整个程序的运行（`menu`的作用即`main`函数）；
> - 首先，对重复使用次数高的操作函数化，包括：
>   - 矩阵的建立；
>   - 菜单的打印；
>   - 矩阵的打印；
> - 使用`menu`调用`menu_print`打印菜单，读取用户输入，并且执行相应的功能。



#### [功能描述]

| 功能           | 复杂度分析                                | 功能说明                                                     |
| -------------- | ----------------------------------------- | ------------------------------------------------------------ |
| Add_mat        | $\Theta(n\times m)$                       | 输入2个矩阵($n\times m$)，相加后输出。                       |
| Num_multi_mat  | $\Theta(n\times m)$                       | 输入1个矩阵($n\times m$)和1个数，输出相乘后的结果。          |
| Trans          | $\Theta\left(n\times m-\min(n, m)\right)$ | 输入1个矩阵($n\times m$)，输出转置后结果(对角线的元素不用动) |
| Dot            | $\Theta(m\times n\times k)$               | 输入2个矩阵($n\times m, m\times k$)，输出其点积后的结果。    |
| Hadamard_multi | $\Theta(m\times n)$                       | 输入2个矩阵($n\times m$)，对应相乘后输出。                   |
| Conv2d         | $\Theta(\frac{n-f}{s}+1)$(单边情况)       | filter边长f，步长s，矩阵边长n，输出卷积后的结果。            |
| Conv2d_apply   | 同上                                      | 卷积的应用，输出6张图片，其中五张为卷积后的                  |

## 在实验过程中遇到的问题及解决方法

#### 可能会遇到的一些问题

- 矩阵之间的size对应关系错误而造成的程序奔溃。

> 这些问题可以通过对输入进行检查解决。

- 没有注意图片的多通道问题，而造成filter卷积后的图像只被处理了1/3。

> `cv::imread`函数默认读入RBG图片，只需要设置`cv::imread(PATH, 0)`即可解决。

## 心得体会

#### [关于本次编程的心得体会]

本次编程我认为我最大的收获是学会了如何规范地写出一份干净清晰的代码。同时考虑了使用者的感受，设计了一系列友好用户提示和输入检查的方法。

#### [结合自身经历关于Convolution（卷积）的心得体会]

#### 卷积的直观理解

![å·ç§¯ç¥ç"ç½ç"](http://dataunion.org/wp-content/uploads/2015/03/6.gif)

> Reference: [[1]](https://www.cnblogs.com/nsnow/p/4562308.html).

#### 卷积的意义

##### 和“那个”卷积不一样

在概率论中我们学到的卷积是这样的：
$$
\int_{-\infin}^{+\infin}f(t)g(x-t)dt
$$
上式是对于连续情况而言，而对于矩阵的二维卷积真正应该是怎么样呢？
$$
C_{a, b} = \sum_{i=0}^{m_f-1}\sum_{j=0}^{n_f-1}M_{(a+m_f-i),(b+n_f-j)}\times f_{i,j}
$$
![img](http://upload-images.jianshu.io/upload_images/2256672-8d30f15073885d7b.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/640)

> Reference: [[2]](https://www.zybuluo.com/hanbingtao/note/485480)
>
> 图片没问题，但这个网站上的公式我认为作者写错了。

也就是说，是反过来的。

问题来了，为什么我们在真正做卷积的时候，没有反过来呢？

因为，假设<u>按照上面的方式</u>我们设计了一个卷积核$F$，然后我们要又按不翻转的方式设计一个卷积核$F'$，然后将他们运用于图像后，我们能得到相同的结果，那么这两个卷积核有什么关系呢？没错，就是翻转180度的关系，由旋转矩阵：
$$
F=F'\times M(\pi)\\
其中M(\pi)=
\left[
	\begin{aligned}
    &cos(\pi)\ &-sin(\pi)\\
    &sin(\pi)\ &cos(\pi)
    \end{aligned}
\right]
$$
既然效果能一样，那么这个旋转矩阵就是多余的，平时我们就用$F'$就好了。

##### 图像处理

在传统图像处理中，卷积操作充分了利用了像素的局部关系，卷积的结果并不是单个像素的运算的结果，而是一片区域内（局部感受野）的像素的提取结果。在传统图像处理的领域，filter一般都是固定的，人为设定的。

##### Convolutional Neural Network

在深度学习的卷积层中也使用了卷积核，它的好处在于能减少参数个数（全连接层的参数大小和图像size的大小一样，而卷积层的参数个数只和filter的size相关），实现权值共享，降低训练的难度。其中的filter的值是由网络训练得到的，卷积核有很强的特征提取能力，比如注明的YOLO v3的成功之处就包括其darknet网络结构除了shortcut和route结构外，都使用的是卷积层，并且用卷积层代替下采样(down pooling)使得特征信息被充分地保留。

#### 关于卷积的一些要素

- 步长stride
- 卷积核大小filter_size
- 留白padding
- 空洞卷积atrous convolutions

> 一般关于dilation可能会有点陌生，其实一张图就能说明白：
>
> ![img](http://img.mp.itc.cn/upload/20170729/14ec9e4b290e451fad28be61170b5dc1_th.jpg)
>
> > Reference: [[3]](https://github.com/vdumoulin/conv_arithmetic).

- 以及最近google团队根据Dropout的思想设计的DropBlock.

> ![å¨è¿éæå¥å¾çæè¿°](http://www.pianshen.com/images/518/d829eccf5f322e7fa0b9145244a6690e.JPEG)
>
> > [DropBlock: A regularization method for convolutional networks](https://arxiv.org/abs/1810.12890v1)



> 声明：本文所有公式和文字均由本人使用`markdown`和`LaTeX`编写。

## Source code

```c++
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
	kernel[4] = (cv::Mat_<char>(5, 5) << 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -4, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0);
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
```

