#pragma once

#include "setting.h"
#include "global_state.h"
#include "game_node.h"
#include "main_node.h"
#include "stop_node.h"

#include <graphics.h>
#include <thread>
#include <mmsystem.h>


#pragma comment(lib,"winmm.lib")


using namespace std;

static auto $$game_init$$ = []()
{

	mciSendString("open music/GAME1.mp3 alias g1", NULL, 0, NULL);
	mciSendString("open music/GAME2.mp3 alias g2", NULL, 0, NULL);
	mciSendString("open music/GAME3.mp3 alias g3", NULL, 0, NULL);
	mciSendString("open music/GAME4.mp3 alias g4", NULL, 0, NULL);
	mciSendString(TEXT("setaudio g1 volume to 2000"), NULL, 0, NULL);
	mciSendString(TEXT("setaudio g2 volume to 2000"), NULL, 0, NULL);
	mciSendString(TEXT("setaudio g3 volume to 2000"), NULL, 0, NULL);
	mciSendString(TEXT("setaudio g4 volume to 2000"), NULL, 0, NULL);
	
	game_property::valid_map.fill(game_property::nullkey);
	for (size_t i = 0; i < strlen(game_property::keys); ++i)
		game_property::valid_map[game_property::keys[i]] = i;
	
	thread listener_thread(global::listen);
	thread bkm_music_thread([]() {
		mciSendString("open music/back_ground.mp3 alias bkg", NULL, 0, NULL);
		mciSendString(TEXT("setaudio bkg volume to 70"), NULL, 0, NULL);
		while (true)
		{
			mciSendString("play bkg from 0 wait", NULL, 0, NULL);
		}
		mciSendString("close bkg", NULL, 0, NULL);
	});

	thread listen_music_thread([]() {
		SLEEP(1000);
		mciSendString("open music/attack1.mp3 alias a1", NULL, 0, NULL);
		mciSendString("open music/attack2.mp3 alias a2", NULL, 0, NULL);
		mciSendString("open music/attack3.mp3 alias a3", NULL, 0, NULL);
		mciSendString("open music/attack4.mp3 alias a4", NULL, 0, NULL);
		mciSendString("open music/lrznb.mp3 alias nb", NULL, 0, NULL);

		mciSendString("open music/npc.mp3 alias n1", NULL, 0, NULL);
		mciSendString("open music/npc2.mp3 alias n2", NULL, 0, NULL);
		mciSendString("open music/npc3.mp3 alias n3", NULL, 0, NULL);
		mciSendString(TEXT("setaudio n1 volume to 200"), NULL, 0, NULL);
		mciSendString(TEXT("setaudio n2 volume to 200"), NULL, 0, NULL);
		mciSendString(TEXT("setaudio n3 volume to 200"), NULL, 0, NULL);

		mciSendString("open music/hurt1.mp3 alias h1", NULL, 0, NULL);
		mciSendString("open music/hurt2.mp3 alias h2", NULL, 0, NULL);

		mciSendString(TEXT("setaudio nb volume to 2000"), NULL, 0, NULL);

		random_device rd;
		default_random_engine e(rd());
		uniform_int_distribution<int> ud(1, 4);
		while (true)
		{
			switch (global::music_id())
			{
			case ADD:
				switch (e() % 5)
				{
				case 0:
					mciSendString("play a1 from 0 wait", NULL, 0, NULL);
					break;
				case 1:
					mciSendString("play a2 from 0 wait", NULL, 0, NULL);
					break;
				case 2:
					mciSendString("play a3 from 0 wait", NULL, 0, NULL);
					break;
				case 3:
					mciSendString("play a4 from 0 wait", NULL, 0, NULL);
					break;
				default:
					mciSendString("play nb from 0 wait", NULL, 0, NULL);
					break;
				}
				break;
			case SUB:
				if (e() % 2)
					mciSendString("play h1 from 0 wait", NULL, 0, NULL);
				else
					mciSendString("play h2 from 0 wait", NULL, 0, NULL);
				break;
			default:
				if (e()%100 < 1)
				{
					switch (e() % 3)
					{
					case 0:
						mciSendString("play n1 from 0 wait", NULL, 0, NULL);
						break;
					case 1:
						mciSendString("play n2 from 0 wait", NULL, 0, NULL);
						break;
					default:
						mciSendString("play n3 from 0 wait", NULL, 0, NULL);
						break;
					}
					SLEEP(1000);
				}
				break;
			}
			global::music_id() = NONE;
			SLEEP(200);
		}
	});

	listener_thread.detach(); // Just continue;
	bkm_music_thread.detach();
	listen_music_thread.detach();
	return nullptr;
}();

namespace game
{
	void run() noexcept
	{// MAIN LOGIC;
		loadimage(&im_manager::game_n_base, _T("imsrc/GAME.png"));
		loadimage(&im_manager::stop_n_base, _T("imsrc/STOP.png"));
		loadimage(&im_manager::main_n_base, _T("imsrc/MAIN.png"));

		loadimage(&im_manager::icons[obj_type::LRZ - 1][0], _T("imsrc/lrzL.png"));
		loadimage(&im_manager::icons[obj_type::LRZ - 1][1], _T("imsrc/lrzR.png"));
		loadimage(&im_manager::icons[obj_type::NORMAL - 1][0], _T("imsrc/ynwg.jpg"));
		loadimage(&im_manager::icons[obj_type::NORMAL - 1][1], _T("imsrc/ynwg2.jpg"));
		loadimage(&im_manager::icons[obj_type::XIAOMING - 1][0], _T("imsrc/wjz.jpg"));
		loadimage(&im_manager::icons[obj_type::XIAOMING - 1][1], _T("imsrc/wjz2.jpg"));

		main_node main_n;
		game_node game_n;
		stop_node stop_n;

		initgraph(game_property::GUI_xlen, game_property::GUI_ylen);
		setbkcolor(RGB(20, 20, 25));													// 重置背景色
		cleardevice();
		
		while (true)
		{
			if (global::request_restart())
			{// 处理重新开启游戏的请求
				global::request_restart() = false;
				game_n.end();
				global::state() = global::states::GAME;
			}

			switch (global::state())
			{
			case global::states::MAIN:
				if (global::gaming())
					game_n.end();
				main_n.execute();
				break;
			case global::states::GAME:
				if (!global::gaming())
					game_n.start();
				game_n.execute();
				break;
			case global::states::OVER:
				quick_exit(0);
				break;
			case global::states::STOP:
				stop_n.execute();
				break;
			}
				
		}

		closegraph();
	}
}