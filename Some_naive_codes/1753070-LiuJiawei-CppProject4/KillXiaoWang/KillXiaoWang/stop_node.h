#pragma once

class stop_node
{
public:
	void execute() noexcept
	{
		// STOP ->
		//                 1.  回到游戏            GAME
		//                 2.  重新开始游戏         GAME-RESTART
		//                 3. 退出游戏，回到主界面  MAIN
		constexpr uint8_t GOTO_GAME    = '1';
		constexpr uint8_t GOTO_RESTART = '2';
		constexpr uint8_t GOTO_MAIN    = 27;

		draw();
		$(global::state())

		do {
			SLEEP(game_property::listen_pause);
		}while (global::now_key() != GOTO_GAME && global::now_key() != GOTO_MAIN && global::now_key() != GOTO_RESTART);
		
		if (global::now_key() == GOTO_RESTART)
			global::request_restart() = true;
		else
			global::state() = global::now_key() == GOTO_MAIN ? global::states::MAIN : global::states::GAME;
		$(global::state());
	}
private:
	void draw() noexcept
	{
		putimage(0, 0, &im_manager::stop_n_base);
	}
};
