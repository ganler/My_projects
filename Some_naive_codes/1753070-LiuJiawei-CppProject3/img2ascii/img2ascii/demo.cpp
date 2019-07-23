#include "PicReader.hpp"
#include "FastPrinter.hpp"
#include <stdio.h>
#include <iostream>
#include <string>
//#include <urlmon.h>
#include "array.hpp"

//#pragma comment(lib, "urlmon.lib")

#include<mmsystem.h>

#pragma comment(lib,"winmm.lib")


void imshow(const char* picname, int pix_sz = 1)
{
	PicReader imread;
	BYTE *data = nullptr;
	UINT x, y;

	// Pure black
	imread.readPic(picname);
	imread.testReader(data, x, y);
	pix_sz = ( x <= 80 ) ? 2 : pix_sz;

	Array img(data, y, x, 4);
	constexpr size_t tms = 2;
	img.to_ascii();

	WORD* colorBuffer = new WORD[img.size() / 2];
	FastPrinter printer(2 * img.cols(), img.rows(), pix_sz);
	SMALL_RECT drawArea{ 0, 0, img.cols() * 2, img.rows() };

	for (int i = 0; i < tms * img.size() / 4; i++)
		colorBuffer[i] = fp_color::f_black | fp_color::b_l_white;
	
	printer.cleanSrceen();
	printer.setData((char*)img.data(), colorBuffer, drawArea);
	printer.draw(true);

	delete[] colorBuffer;
	printf("Press enter to continue...");
	getchar();
}

void play_demo()
{
	
	PicReader imread;
	BYTE *data = nullptr;
	UINT x, y;

	// Pure black
	imread.readPic("imsrc\\1.jpg");
	imread.testReader(data, x, y);

	Array img(data, y, x, 4);
	constexpr size_t tms = 2;
	img.to_ascii();

	WORD* colorBuffer = new WORD[img.size() / 2];
	FastPrinter printer(2 * img.cols(), img.rows(), 2);
	SMALL_RECT drawArea{ 0, 0, img.cols() * 2, img.rows() };
	Sleep(500);
	mciSendString("open demo.mp3 alias aa", NULL, 0, NULL);//alias后面为设备名称
	mciSendString(TEXT("play aa"), NULL, 0, NULL);

	constexpr uint8_t colors[] = { fp_color::f_red, fp_color::f_blue, fp_color::f_black, fp_color::f_purple, fp_color::f_gray, fp_color::f_green };
	uint8_t fc = 0;
	for (int i = 1, cnt = 0; i <= 155; i++) {

		imread.readPic(("imsrc\\" + std::to_string(i) + ".jpg").c_str());
		imread.testReader(std::ref(img.data()), x, y);
		img.to_ascii();

		if (++cnt % 100 == 0)
			fc = colors[(cnt / 100) % 6];

		for (int i = 0; i < tms * img.size() / 4; i++)
			colorBuffer[i] = fc | fp_color::b_l_white;
		Sleep(35);
		printer.cleanSrceen();
		printer.setData((char*)img.data(), colorBuffer, drawArea);
		printer.draw(true);
	}

	delete[] colorBuffer;
	mciSendString("play aa wait", NULL, 0, NULL);
	mciSendString("close aa", NULL, 0, NULL);
	printf("Press enter to continue...");
	getchar();
}

int main() {

	{// Images test.
		imshow("classic_picture/airplane.jpg");
		imshow("classic_picture/baboon.jpg");
		imshow("classic_picture/barbara.jpg");
		imshow("classic_picture/cameraman.jpg");
		imshow("classic_picture/compa.png");
		imshow("classic_picture/lena.jpg");
		imshow("classic_picture/lena1.jpg");
		imshow("classic_picture/milkdrop.jpg");
		imshow("classic_picture/peppers.jpg");
		imshow("classic_picture/woman.jpg");
	}

	{
		play_demo();
	}
}