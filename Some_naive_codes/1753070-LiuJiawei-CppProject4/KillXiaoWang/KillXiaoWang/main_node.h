#pragma once

/*
INPUT 说明：
 1. 开始游戏 -> goto_game；
 2. 结束游戏 -> goto_over；
*/


class main_node
{
public:
	void execute() noexcept
	{
		// MAIN ->
		//            1. GAME
		//            2. OVER
		constexpr uint8_t GOTO_GAME = '1';
		constexpr uint8_t GOTO_OVER = '2';

		draw();
		$(global::state())

		do {
			SLEEP(game_property::listen_pause);
		} while (global::now_key() != GOTO_GAME && global::now_key() != GOTO_OVER);

		global::state() = (global::now_key() == GOTO_GAME) ? global::states::GAME : global::states::OVER;
		$(global::state());
	}
private:
	void draw() noexcept
	{
		putimage(0, 0, &im_manager::main_n_base);
	}
};