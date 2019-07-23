#pragma once

#define SLEEPFOR(x) { std::this_thread::sleep_for(std::chrono::milliseconds((x))); }
#define GAME_PAUSE  80
#define COLS        700
#define ROWS        500

#include <chrono>
#include <thread>
#include <graphics.h>
#include <conio.h>
#include <thread>
#include <string>
#include <fstream>
#include <iostream>
#include <sstream>

#include "game_map.hpp"

using namespace std;


class Information
{
public:

	void update_info()
	{
		std::string news;
		static random_device rd;
		static mt19937 engine{ rd() };
		static uniform_int_distribution<uint16_t> u(0, 9);
		static RECT news_impl = { 0, 355, 190, 450 };

		drawtext(/* 手动清屏 */
			"                                       \n                                       \n                                       \n                                       ",
			&news_impl, DT_CENTER
		);
		switch (u(engine))
		{
		case 0:
			news = "吾日三省吾身，\n李润中为什么那么强，\n王维斯为什么那么爱脉弱，\n刘佳伟为什么那么菜";
			break;
		case 1:
			news = "报告助教：李润中nb";
			break;
		case 2:
			news = "每当你遇到困难\n大喊一句lrznb就完事了";
			break;
		case 3:
			news = "这条蛇穿的怕是品如的衣服";
			break;
		case 4:
			news = "本程序GUI是拿EasyX画的\n为了游戏流畅我使用了\ndeque以及\n只置换新旧节点来加速";
			break;
		case 5:
			news = "放飞版中\n吃蓝果子会中毒\n使控制反向\n粉果子解读且+10分\n";
			break;
		case 6:
			news = "设置Release编译模式\n可得到更加流畅的\n游戏体验哦";
			break;
		case 7:
			news = "你玩贪吃蛇有点像蔡徐坤";
			break;
		case 8:
			news = "你可以使用awsd\n也可以直接上下左右";
			break;
		case 9:
			news = R"(抵制不良游戏 拒绝盗版游戏
注意自我保护 谨防受骗上当
适度游戏益脑 过度游戏伤身
合理安排时间 享受健康生活)";
			break;
		}
		drawtext(news.c_str(), &news_impl, DT_CENTER);
	}

	void draw(int score, int highest) noexcept
	{
		LOGFONT f;
		gettextstyle(&f);
		f.lfHeight = 18.5;
		f.lfWeight = 4;
		_tcscpy_s(f.lfFaceName, _T("微软雅黑"));
		f.lfQuality = ANTIALIASED_QUALITY;
		settextstyle(&f);


		static uint64_t counter;
		RECT score_impl = { 10, 60, 190, 100 };
		drawtext((to_string(score) + " / " + to_string(highest) + " (历史最高纪录)").c_str(), &score_impl, DT_CENTER);
		if (counter++ % 20 == 0)
			update_info();
	}

	void init(mode fm) noexcept
	{
		LOGFONT f;
		gettextstyle(&f);
		f.lfHeight = 24;
		f.lfWeight = 6;
		_tcscpy_s(f.lfFaceName, _T("微软雅黑"));
		f.lfQuality = ANTIALIASED_QUALITY;
		settextstyle(&f);

		RECT score_zone = { 10, 10, 190, 50 };
		drawtext(">>> Scores", &score_zone, DT_CENTER);

		std::string level_str = ">>> Level\n\n";
		level_str += fm == mode::unknown ? "{ 请选择 }" : (fm == mode::easy) ? "{ 入门版 }" : (fm == mode::mid ? "{ 进阶版 }" : (fm == mode::fly ? "{ 放飞版 }" :"{ 高级版 }"));

		RECT level_zone = { 10, 160, 190, 250 };
		drawtext(level_str.c_str(), &level_zone, DT_CENTER);

		RECT news_zone = { 10, 290, 190, 330 };
		drawtext(">>> News", &news_zone, DT_CENTER);
	}
};

class game
{
public:
    void run()
	{
		update_history_score();
		while (menu())
		{
			thread t([&]() {do
			{
				SLEEPFOR(GAME_PAUSE);
				draw();
			} while (gaming = m_map.snake_next_step() && !strong_ext); } );

			while (gaming)
				if (_kbhit())
					switch (_getch())
					{
					case 72: 
					case 'w':
						m_map.dir = direction::north; break;
					case 80: 
					case 's':
						m_map.dir = direction::south; break;
					case 75: 
					case 'a':
						m_map.dir = direction::west; break;
					case 77: 
					case 'd':
						m_map.dir = direction::east; break;
					case 27:
						strong_ext = true;
						break;
					default:
						break;
					}

			t.join();
			update_history_score();
		}
		
	}
public:
	~game()
	{
		closegraph();
	}
	bool menu()
	{// 选取模式
		initgraph(COLS, ROWS);

		setlinestyle(PS_SOLID, 3);
		setbkcolor(WHITE);
		settextcolor(BLACK);


		cleardevice();
		
		LOGFONT f;
		gettextstyle(&f);
		f.lfHeight = 24;
		f.lfWeight = 6;
		_tcscpy_s(f.lfFaceName, _T("微软雅黑"));
		f.lfQuality = ANTIALIASED_QUALITY;
		settextstyle(&f);
		RECT menu = { 10, 10, COLS, ROWS};
		drawtext("请选择模式：\n  a: 入门版本\n  b: 进阶版本 \n  c: 高级版本\n  d: 放飞版本\n\n [ESC]: 退出游戏", &menu, DT_CENTER);
		
		

		bool not_inp = !(gmode == mode::hard && m_map.lifes < 5 && m_map.lifes > 0);


		while (not_inp)
		{
			not_inp = false;
			switch (_getch())
			{
			case 'a':
				gmode = mode::easy;
				break;
			case 'b':
				gmode = mode::mid; 
				break;
			case 'c':
				gmode = mode::hard; 
				break;
			case 'd':
				gmode = mode::fly; 
				break;
			case 27:
				quick_exit(0);
			default:
				not_inp = true;
				break;
			}
		}

		restart();
		return true;
	}
	void draw()
	{
		m_map.draw();
		history_highest_score = max(history_highest_score, m_map.scores);
		m_info.draw(m_map.scores, history_highest_score);
	}
	void restart()
	{
		
		strong_ext = false;
		history_highest_score = history[gmode];
		gaming = true;
		m_map.init(gmode);
		m_info.init(gmode);
	}
public:
	// Utils
	void update_history_score()
	{
		constexpr char filename[] = "record.txt";
		ifstream file(filename);
		if (file.is_open() && file.peek() != ifstream::traits_type::eof())
		{
			for (int i = 0; i < 4; ++i)
				file >> history[i];

			history[gmode] = max(history[gmode], history_highest_score);

			ofstream create_file;
			create_file.open(filename);
			file.clear();
			for (int i = 0; i < 4; ++i)
				create_file << history[i] << ' ';
			create_file.close();
			file.close();
		}
		else
		{
			ofstream create_file;
			create_file.open(filename);
			for (int i = 0; i < 4; ++i)
				create_file << 0 << ' ';
			create_file.close();
		}
	}

private:
	game_map     m_map;
	int          history_highest_score;
	int          history[4];
	mode         gmode;
	Information  m_info;
	bool         gaming;
	bool         strong_ext;
};