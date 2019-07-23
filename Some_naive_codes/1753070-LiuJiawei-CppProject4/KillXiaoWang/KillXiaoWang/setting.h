#pragma once

#include <cstdint>
#include <iostream>
#include <array>
#include <thread>
#include <chrono>
#include <type_traits>

#include <graphics.h>

#define DEBUG 1
#define $(x) {\
  if(DEBUG) \
	if( std::is_integral_v<decltype(x)>)\
		std::cout << "LINE" << __LINE__ << ": DEBUGGING  " << #x << " == " << (int64_t)(x) << '\n'; \
	else\
		std::cout << "LINE" << __LINE__ << ": DEBUGGING  " << #x << " == " << (x) << '\n';}
#define SLEEP(x) { std::this_thread::sleep_for(std::chrono::milliseconds(x)); }
#define LOG(x) { if(DEBUG) std::cout << "LOGGING :  " << (x) << '\n'; }


namespace game_property
{
	
	// TIME
	constexpr uint32_t listen_pause = 25;
	constexpr uint32_t time_slice   = 200;
	constexpr uint32_t draw_pause   = 25;

	// Filename
	constexpr char score_file[]     = "record.txt";

	// valid key
	std::array<uint8_t, 256> valid_map;
	constexpr char keys[]           = { "qweasdzxc" };
	constexpr char nullkey          = 'n';

	// map
	constexpr int GUI_xlen          = 720;
	constexpr int GUI_ylen          = 480;

	// icon image
	constexpr int icon_sz           = 158;
	constexpr int icon_gap          = 3;
	constexpr int icon_shift        = GUI_xlen - GUI_ylen;
	constexpr std::array<POINT, 9> coordinates{ {
		{icon_shift + icon_gap, icon_gap}, {icon_shift + icon_gap * 2 + icon_sz, icon_gap}, {icon_shift + icon_gap*3 + icon_sz*2, icon_gap},
		{icon_shift + icon_gap, icon_gap*2+icon_sz}, {icon_shift + icon_gap * 2 + icon_sz, icon_gap*2+ icon_sz}, {icon_shift + icon_gap * 3 + icon_sz * 2, icon_gap * 2 + icon_sz},
		{icon_shift + icon_gap, icon_gap * 3 + icon_sz*2}, {icon_shift + icon_gap * 2 + icon_sz, icon_gap *3 + icon_sz*2}, {icon_shift + icon_gap * 3 + icon_sz * 2, icon_gap * 3 + icon_sz*2}
		} };
}
