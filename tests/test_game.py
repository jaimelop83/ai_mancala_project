# Tests for game logic
# tests/test_game.py

import unittest
from game import Mancala

class TestMancala(unittest.TestCase):

    def setUp(self):
        self.game = Mancala()

    def test_initial_state(self):
        self.assertEqual(self.game.board, [4]*6 + [0] + [4]*6 + [0])
        self.assertEqual(self.game.current_player, 1)

    def test_get_legal_moves_player1(self):
        self.assertEqual(self.game.get_legal_moves(), [0, 1, 2, 3, 4, 5])

    def test_get_legal_moves_player2(self):
        self.game.current_player = 2
        self.assertEqual(self.game.get_legal_moves(), [7, 8, 9, 10, 11, 12])

    def test_apply_move_increments_correct_pits(self):
        # Pick pit 2 (index 2) with 4 stones, expect to land one in pits 3, 4, 5, and store
        self.game.board = [4, 4, 4, 4, 4, 4, 0,
                           4, 4, 4, 4, 4, 4, 0]
        self.game.apply_move(2)
        expected = [4, 4, 0, 5, 5, 5, 1,
                    4, 4, 4, 4, 4, 4, 0]
        self.assertEqual(self.game.board, expected)

    def test_extra_turn_if_landing_in_own_store(self):
        # Set up so last stone lands in player 1's store
        self.game.board = [0, 0, 0, 0, 0, 1, 0,
                           4, 4, 4, 4, 4, 4, 0]
        self.game.current_player = 1
        self.game.apply_move(5)
        self.assertEqual(self.game.current_player, 1)

    def test_switch_player_when_not_landing_in_store(self):
        self.game.board = [0, 0, 0, 0, 0, 2, 0,
                           4, 4, 4, 4, 4, 4, 0]
        self.game.current_player = 1
        self.game.apply_move(5)
        self.assertEqual(self.game.current_player, 2)

    def test_capture_logic(self):
        # Setup: Player 1 has a single stone in pit 1 (index 1)
        # and pit 1 is empty; pit 10 (opposite) has 4 stones
        self.game.board = [0, 1, 0, 0, 0, 0, 0,
                           0, 0, 0, 4, 0, 0, 0]
        self.game.current_player = 1
        self.game.apply_move(1)  # Drops stone into index 2 (empty), should capture from index 10
        self.assertEqual(self.game.board[6], 5)  # 1 landed + 4 captured
        self.assertEqual(self.game.board[2], 0)
        self.assertEqual(self.game.board[10], 0)

    def test_game_over_sweeps_remaining_stones(self):
        # Only one move left for player 1
        self.game.board = [0, 0, 0, 0, 0, 1, 0,
                           1, 0, 0, 0, 0, 0, 5]
        self.game.current_player = 1
        self.game.apply_move(5)  # Ends game
        self.assertTrue(self.game.is_game_over())
        self.assertEqual(self.game.board[6], 1)   # 1 stone left goes to P1 store
        self.assertEqual(self.game.board[13], 6)  # P2 store: 5 existing + 1 swept

    def test_illegal_move_from_empty_pit(self):
        self.game.board[0] = 0
        with self.assertRaises(ValueError):
            self.game.apply_move(0)

    def test_get_score_returns_correct_values(self):
        self.game.board[6] = 10
        self.game.board[13] = 7
        score = self.game.get_score()
        self.assertEqual(score, (10, 7))


if __name__ == '__main__':
    unittest.main()

'''
	•	Initial Game State
	•	Validates the board starts with 4 stones in each pit and 0 in stores.
	•	Checks that Player 1 starts first.
	•	Legal Moves
	•	Ensures Player 1 can choose from pits 0 5 if they contain stones.
	•	Ensures Player 2 can choose from pits 7 12 if they contain stones.
	•	Invalid Moves
	•	Verifies that selecting an empty pit raises a ValueError.
	•	Move Application (Non-Capture)
	•	Tests normal pit selection and stone distribution.
	•	Confirms that if the last stone lands in the current players store, they get another turn.
	•	Turn Switching
	•	Confirms turn changes if the last stone doesnt land in the players store.
	•	Capture Mechanic
	•	Validates capturing when the last stone lands in an empty pit on the players side opposite a non-empty opponent pit.
	•	Game Over Logic
	•	Ensures when one sides pits are empty:
	•	Remaining stones are swept into the opponents store.
	•	Final score is calculated correctly.
	•	is_game_over() returns True.
    '''