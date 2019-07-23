#pragma once

#include "setting.h"

#include <conio.h>
#include <cstdint>
#include <iostream>
#include <vector>
#include <fstream>

using namespace std;

enum obj_type : uint8_t
{
	EMPTY    = 0,
	NORMAL   = 1,
	LRZ      = 2,
	XIAOMING = 3
};

enum music_type : uint8_t
{
	ADD,
	SUB,
	NONE
};

struct global
{
	global(...) = delete;

	enum class states : uint8_t
	{
		MAIN    = 'M',
		GAME    = 'G',
		STOP    = 'S',
		OVER    = 'O'
	};

	friend ostream& operator << (ostream& os, states s)
	{// For debug.
		switch (s)
		{
		case states::MAIN:
			os << "MAIN";
			break;
		case states::GAME:
			os << "GAME";
			break;
		case states::STOP:
			os << "STOP";
			break;
		case states::OVER:
			os << "OVER";
			break;
		}
		return os;
	}
	
	static music_type& music_id()
	{
		static music_type g_music_id = NONE;
		return g_music_id;
	}

	static bool& request_restart()
	{
		static bool g_to_restart = false;
		return g_to_restart;
	}

	static bool& gaming()
	{
		static bool g_gaming = false;
		return g_gaming;
	}

	static states& state()
	{
		static states g_state = states::MAIN;
		return g_state;
	}

	static void listen()
	{
		while (state() != states::OVER)
		{
			SLEEP(game_property::listen_pause);
			if (_kbhit()) { now_key_pr() = _getch(); }
			else now_key_pr() = 0;
		}
	}

	const static uint8_t now_key()
	{
		return now_key_pr();
	}

private:
	static uint8_t& now_key_pr()
	{
		constexpr uint8_t NO_INP = 0;
		static uint8_t g_input = NO_INP;
		return g_input;
	}
};

struct score_helper
{
	// [score]
	score_helper(...) = delete;

	static void update(uint16_t level, uint16_t s) noexcept
	{
		// LOG("更新最高分\n!");
		constexpr uint64_t map[] = { 0xffffff00 , 0xffff00ff , 0xff00ffff , 0x00ffffff };
		if (static_cast<uint16_t>(score_pr() >> ((level - 1) * 16)) < s)
			( score_pr() &= (map[level - 1]) ) |= (s << ((level - 1) * 16));
	}
	static void write_score() noexcept
	{
		// LOG("写文件！\n");
		static ofstream file(game_property::score_file);
		file.clear();
		file << score_pr();
		file.close();
	}
	static void read_score() noexcept
	{
		// LOG("读文件！\n");
		static ifstream file(game_property::score_file);
		if (file.is_open() && file.peek() != ifstream::traits_type::eof())
			file >> score_pr();
	}
	static uint64_t max_score(uint16_t level) noexcept
	{
		return static_cast<uint16_t>(score_pr() >> ((level - 1) * 16));
	}
private:
	static uint64_t& score_pr() noexcept
	{
		static uint64_t history_score = 0;
		return history_score;
	}
};

namespace im_manager
{
	IMAGE icons[3][2]{  {IMAGE(game_property::icon_sz, game_property::icon_sz), IMAGE(game_property::icon_sz, game_property::icon_sz)},
						{IMAGE(game_property::icon_sz, game_property::icon_sz), IMAGE(game_property::icon_sz, game_property::icon_sz)}, 
						{IMAGE(game_property::icon_sz, game_property::icon_sz), IMAGE(game_property::icon_sz, game_property::icon_sz)} };

	IMAGE game_n_base(game_property::GUI_xlen, game_property::GUI_ylen);
	IMAGE main_n_base(game_property::GUI_xlen, game_property::GUI_ylen);
	IMAGE stop_n_base(game_property::GUI_xlen, game_property::GUI_ylen);

}
