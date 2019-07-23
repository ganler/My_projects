/******************************************************************
* ！注意！                                                         *
* 本头文件中为你封装了WinAPI中有关Console绘制的底层函数，可以帮助你快  *
* 速绘制你想要的输出，效率比printf+cls高出很多。                     *
* 函数使用详见demo.cpp中的几个示例。                                *
******************************************************************/
#ifndef FAST_PRINTER_H
#define FAST_PRINTER_H
#include <windows.h>

/******************************************************************
*  TO-DO:                                                         *
*                                                                 *
*  本文件你可以自由进行修改，如将其中的一些接收参数设置为你实现的Array  *
*  或为了配合你的实现进行一些便携化成员函数的编写等，甚至自己重新实现   *
*  一个更高效的。                                                  *
*                                                                 *
******************************************************************/

namespace fp_color {
	// f is the foreground color b is the background color
	// console color format: (f | b)
	const SHORT f_black = 0;
	const SHORT f_blue = 0x0001;
	const SHORT f_green = 0x0002;
	const SHORT f_aqua = 0x0003;
	const SHORT f_red = 0x0004;
	const SHORT f_purple = 0x0005;
	const SHORT f_yellow = 0x0006;
	const SHORT f_white = 0x0007;
	const SHORT f_gray = 0x0008;
	const SHORT f_l_blue = 0x0009;
	const SHORT f_l_green = 0x000A;
	const SHORT f_l_aqua = 0x000B;
	const SHORT f_l_red = 0x000C;
	const SHORT f_l_purple = 0x000D;
	const SHORT f_l_yellow = 0x000E;
	const SHORT f_l_white = 0x000F;

	const SHORT b_black = 0;
	const SHORT b_blue = 0x0010;
	const SHORT b_green = 0x0020;
	const SHORT b_aqua = 0x0030;
	const SHORT b_red = 0x0040;
	const SHORT b_purple = 0x0050;
	const SHORT b_yellow = 0x0060;
	const SHORT b_white = 0x0070;
	const SHORT b_gray = 0x0080;
	const SHORT b_l_blue = 0x0090;
	const SHORT b_l_green = 0x00A0;
	const SHORT b_l_aqua = 0x00B0;
	const SHORT b_l_red = 0x00C0;
	const SHORT b_l_purple = 0x00D0;
	const SHORT b_l_yellow = 0x00E0;
	const SHORT b_l_white = 0x00F0;
}

class FastPrinter {
public:
	FastPrinter(DWORD, DWORD);
	FastPrinter(DWORD, DWORD, WORD);
	~FastPrinter();

	void setData(const char*, const WORD*);
	void setData(const char*, const WORD*, SMALL_RECT);
	void setRect(SMALL_RECT, const char, const WORD);
	void fillRect(SMALL_RECT, const char, const WORD);
	void setText(COORD, const char*, const WORD, const WORD);
	void setText(COORD, const char*, const WORD);
	void setText(COORD, const char*);

	void cleanSrceen();
	void draw(bool);
private:
	HANDLE hOutput, hOutBuf, hTmpBuf;
	COORD coordBufSize;
	COORD coordBufCoord;
	DWORD bytes = 0;
	DWORD sizeX, sizeY;

	char* dataGrid;
	WORD* colorGrid;
	CHAR_INFO* outputGrid;
	SMALL_RECT srctWriteRect;

	void initDrawer();
	void _setFontSize(const WORD);
	void _destroy();

	void _swapBuf();
	void _draw();
	void _drawC();
};

FastPrinter::FastPrinter(DWORD x, DWORD y) :sizeX(x), sizeY(y) {
	initDrawer();
}

FastPrinter::FastPrinter(DWORD x, DWORD y, WORD fontSize) : sizeX(x), sizeY(y) {
	// init with font size
	_setFontSize(fontSize);
	initDrawer();
}

FastPrinter::~FastPrinter() {
	_destroy();
}

void FastPrinter::setData(const char* _in_data, const WORD* _in_color) {
	// copy the data to inner buffer
	memcpy(dataGrid, _in_data, sizeX * sizeY);
	memcpy(colorGrid, _in_color, sizeX * sizeY * sizeof(WORD));
}

void FastPrinter::setData(const char* _in_data, const WORD* _in_color, SMALL_RECT _area) {
	// copy the data to the specified area
	SHORT row = (_area.Right - _area.Left);
	for (WORD _i = _area.Top, i = 0; _i < _area.Bottom; _i++, i++) {
		memcpy(dataGrid + (_i * sizeX + _area.Left), _in_data + (i * row), row);
		memcpy(colorGrid + (_i * sizeX + _area.Left), _in_color + (i * row), row * sizeof(WORD));
	}
}

void FastPrinter::setRect(SMALL_RECT _area, const char _val, const WORD _color) {
	// draw a hollow rectangle
	for (WORD i = _area.Left; i < _area.Right; i++) {
		dataGrid[_area.Top * sizeX + i] = _val;
		dataGrid[(_area.Bottom - 1) * sizeX + i] = _val;

		colorGrid[_area.Top * sizeX + i] = _color;
		colorGrid[(_area.Bottom - 1) * sizeX + i] = _color;
	}

	for (WORD i = _area.Top; i < _area.Bottom; i++) {
		dataGrid[i * sizeX + _area.Left] = _val;
		dataGrid[i * sizeX + _area.Right - 1] = _val;

		colorGrid[i * sizeX + _area.Left] = _color;
		colorGrid[i * sizeX + _area.Right - 1] = _color;
	}
}

void FastPrinter::fillRect(SMALL_RECT _area, const char _val, const WORD _color) {
	// draw a solid rectangle
	SHORT row = (_area.Right - _area.Left);
	for (WORD _i = _area.Top, i = 0; _i < _area.Bottom; _i++, i++) {
		memset(dataGrid + (_i * sizeX + _area.Left), _val, row);
		for (WORD _j = _area.Left; _j < _area.Right; _j++) {
			colorGrid[_i * sizeX + _j] = _color;
		}
	}
}

void FastPrinter::setText(COORD _pos, const char* _val, const WORD _color, const WORD len) {
	// print text with position and color
	// Note: try not to set text with '\n'
	memcpy(dataGrid + (_pos.Y * sizeX + _pos.X), _val, len);
	for (WORD i = _pos.X; i < _pos.X + len; i++) {
		colorGrid[_pos.Y * sizeX + i] = _color;
	}
}

void FastPrinter::setText(COORD _pos, const char* _val, const WORD _color) {
	// print text with position and color but no len
	WORD len = (WORD)strlen(_val);
	memcpy(dataGrid + (_pos.Y * sizeX + _pos.X), _val, len);
	for (WORD i = _pos.X; i < _pos.X + len; i++) {
		colorGrid[_pos.Y * sizeX + i] = _color;
	}
}

void FastPrinter::setText(COORD _pos, const char* _val) {
	// print text with position but no len
	WORD len = (WORD)strlen(_val);
	memcpy(dataGrid + (_pos.Y * sizeX + _pos.X), _val, len);
	for (WORD i = _pos.X; i < _pos.X + len; i++) {
		colorGrid[_pos.Y * sizeX + i] = fp_color::f_l_white;
	}
}

void FastPrinter::_setFontSize(const WORD x) {
	CONSOLE_FONT_INFOEX cfi;
	cfi.cbSize = sizeof(cfi);
	GetCurrentConsoleFontEx(GetStdHandle(STD_OUTPUT_HANDLE), FALSE, &cfi);
	cfi.dwFontSize.X = 0;
	cfi.dwFontSize.Y = x;
	SetCurrentConsoleFontEx(GetStdHandle(STD_OUTPUT_HANDLE), FALSE, &cfi);
}


void FastPrinter::cleanSrceen() {
	memset(dataGrid, 0, sizeX * sizeY);
	memset(colorGrid, 0, sizeX * sizeY * sizeof(WORD));
	memset(outputGrid, 0, sizeX * sizeY * sizeof(CHAR_INFO));
}

void FastPrinter::draw(bool withColor) {
	// flush the whole screen
	if (withColor)_drawC();
	else _draw();
	_swapBuf();
}

void FastPrinter::initDrawer() {
	// init the data buffer
	dataGrid = new char[sizeX * sizeY];
	memset(dataGrid, 0, sizeX * sizeY);

	colorGrid = new WORD[sizeX * sizeY];
	memset(colorGrid, 0, sizeX * sizeY * sizeof(WORD));

	outputGrid = new CHAR_INFO[sizeX * sizeY];
	memset(outputGrid, 0, sizeX * sizeY * sizeof(CHAR_INFO));

	// set the draw area
	srctWriteRect.Top = 0;
	srctWriteRect.Left = 0;
	srctWriteRect.Bottom = (SHORT)(sizeY - 1);
	srctWriteRect.Right = (SHORT)(sizeX - 1);

	// get font size
	CONSOLE_FONT_INFOEX cfi;
	cfi.cbSize = sizeof(cfi);
	GetCurrentConsoleFontEx(GetStdHandle(STD_OUTPUT_HANDLE), FALSE, &cfi);

	// load the external WinAPI Module
	typedef HWND(WINAPI *PROCGETCONSOLEWINDOW)();
	PROCGETCONSOLEWINDOW GetConsoleWindow;
	HMODULE hKernel32 = GetModuleHandleA("kernel32");
	GetConsoleWindow = (PROCGETCONSOLEWINDOW)GetProcAddress(hKernel32, "GetConsoleWindow");

	// get console window handle and move the window to the upper left 
	HWND hwnd = GetConsoleWindow();
	SetWindowPos(hwnd, HWND_TOP, 0, 0, cfi.dwFontSize.X * sizeX, cfi.dwFontSize.Y * sizeY, 0);

	// resize the window
	char cmd_buffer[32] = "mode con: cols=0000 lines=0000";
	cmd_buffer[15] = '0' + (sizeX / 1000 % 10);
	cmd_buffer[16] = '0' + (sizeX / 100 % 10);
	cmd_buffer[17] = '0' + (sizeX / 10 % 10);
	cmd_buffer[18] = '0' + sizeX % 10;

	cmd_buffer[26] = '0' + (sizeY / 1000 % 10);
	cmd_buffer[27] = '0' + (sizeY / 100 % 10);
	cmd_buffer[28] = '0' + (sizeY / 10 % 10);
	cmd_buffer[29] = '0' + sizeY % 10;

	system(cmd_buffer);

	// create output buffer
	hOutBuf = CreateConsoleScreenBuffer(
		GENERIC_WRITE | GENERIC_READ,
		FILE_SHARE_WRITE | FILE_SHARE_READ,
		NULL,
		CONSOLE_TEXTMODE_BUFFER,
		NULL
	);

	hOutput = CreateConsoleScreenBuffer(
		GENERIC_WRITE | GENERIC_READ,
		FILE_SHARE_WRITE | FILE_SHARE_READ,
		NULL,
		CONSOLE_TEXTMODE_BUFFER,
		NULL
	);

	// invisible the cursor
	CONSOLE_CURSOR_INFO cci;
	cci.bVisible = 0;
	cci.dwSize = 1;
	SetConsoleCursorInfo(hOutput, &cci);
	SetConsoleCursorInfo(hOutBuf, &cci);
}

void FastPrinter::_destroy() {
	// clean up memory
	delete[] dataGrid;
	delete[] colorGrid;
	delete[] outputGrid;

	CloseHandle(hOutBuf);
	CloseHandle(hOutput);
}

void FastPrinter::_swapBuf() {
	// core function: display after the data has been set
	hTmpBuf = hOutBuf;
	hOutBuf = hOutput;
	hOutput = hTmpBuf;
}

void FastPrinter::_draw() {
	for (DWORD i = 0; i < sizeY; i++) {
		// draw every line 
		coordBufCoord.Y = (SHORT)i;
		WriteConsoleOutputCharacterA(hOutput, dataGrid + (i * sizeX), sizeX, coordBufCoord, &bytes);
	}
	SetConsoleActiveScreenBuffer(hOutput);
}

void FastPrinter::_drawC() {
	for (DWORD i = 0; i < sizeY; i++) {
		for (DWORD j = 0; j < sizeX; j++) {
			// copy info to CHAR_INFO struct
			// this will draw with color
			outputGrid[i * sizeX + j].Attributes = colorGrid[i * sizeX + j];
			outputGrid[i * sizeX + j].Char.AsciiChar = dataGrid[i * sizeX + j];
		}
	}

	coordBufCoord.X = 0;
	coordBufCoord.Y = 0;
	coordBufSize.X = (SHORT)(sizeX);
	coordBufSize.Y = (SHORT)(sizeY);

	WriteConsoleOutputA(
		hOutput,          // screen buffer to write to 
		outputGrid,       // buffer to copy from 
		coordBufSize,     // col-row size of chiBuffer 
		coordBufCoord,    // top left src cell in chiBuffer 
		&srctWriteRect);  // dest. screen buffer rectangle 
	SetConsoleActiveScreenBuffer(hOutput);
}
/******************************************************************
*  TO-DO END                                                      *
******************************************************************/
#endif