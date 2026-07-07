import unittest

from src.scanner.assembly_planner import plan_test_placement


class AssemblyPlannerTests(unittest.TestCase):
    def test_zhenhong_h2_places_at_row1_col0(self):
        plan = plan_test_placement(role_name="真红", piece_id="H_2", config_dir="config")
        self.assertEqual("真红", plan.role_name)
        self.assertEqual("H_2", plan.piece_id)
        self.assertEqual(1, plan.start_r)
        self.assertEqual(0, plan.start_c)
        self.assertEqual(["H_2", "H_2"], plan.solved_board[1][:2])


if __name__ == "__main__":
    unittest.main()
