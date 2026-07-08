import json
import tempfile
import unittest
from pathlib import Path

from src.app import runtime
from src.scanner.assembly_planner import (
    ASSEMBLY_TEST_ROLE,
    ASSEMBLY_TEST_SHAPE,
    find_first_inventory_drive,
    find_shape_anchor_in_blueprint,
    iter_blueprint_piece_placements,
    plan_assembly,
    plan_full_assembly,
    plan_test_placement,
)
from src.solver.orchestrator import NTEPipelineOrchestrator


class AssemblyPlannerTests(unittest.TestCase):
    def test_zhenhong_h2_places_at_row1_col0(self):
        plan = plan_test_placement(role_name="真红", piece_id="H_2", config_dir="config")
        self.assertEqual("真红", plan.role_name)
        self.assertEqual("H_2", plan.piece_id)
        self.assertEqual(1, plan.start_r)
        self.assertEqual(0, plan.start_c)
        self.assertEqual(["H_2", "H_2"], plan.solved_board[1][:2])

    def test_find_first_inventory_drive(self):
        inventory = [
            {"uid": "a", "shape_id": "V_2"},
            {"uid": "b", "shape_id": "H_2", "sub_stats": {"暴击率%": 2.0}},
            {"uid": "c", "shape_id": "H_2"},
        ]
        drive = find_first_inventory_drive(inventory, "H_2")
        self.assertEqual("b", drive["uid"])

    def test_find_shape_anchor_in_blueprint(self):
        blueprint = [
            ["XX", "XX", "XX", "XX", "XX"],
            ["H_2", "H_2", "0", "0", "0"],
            ["0", "0", "0", "0", "0"],
            ["0", "0", "0", "0", "0"],
            ["0", "0", "0", "0", "0"],
        ]
        piece_matrix = [[1, 1]]
        start_r, start_c = find_shape_anchor_in_blueprint(blueprint, "H_2", piece_matrix)
        self.assertEqual((1, 0), (start_r, start_c))

    def test_plan_assembly_reads_jiuyuan_loadout(self):
        blueprint = [
            ["XX", "XX", "XX", "XX", "XX"],
            ["H_2", "H_2", "0", "0", "0"],
            ["0", "0", "0", "0", "0"],
            ["0", "0", "0", "0", "0"],
            ["0", "0", "0", "0", "0"],
        ]
        equipped_state = {
            ASSEMBLY_TEST_ROLE: {
                "blueprint_layout": blueprint,
                "equipped_drives": [
                    {"uid": "plan_h2", "shape_id": "H_2", "sub_stats": {"暴击率%": 2.0}}
                ],
            }
        }
        inventory = [{"uid": "inv_h2", "shape_id": "H_2", "sub_stats": {"暴击率%": 2.0}}]

        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / "equipped_state.json"
            state_path.write_text(json.dumps(equipped_state, ensure_ascii=False), encoding="utf-8")
            old_user_dir = runtime.USER_CONFIG_DIR
            runtime.USER_CONFIG_DIR = Path(tmp)
            try:
                plan = plan_assembly(
                    role_name=ASSEMBLY_TEST_ROLE,
                    piece_id=ASSEMBLY_TEST_SHAPE,
                    config_dir="config",
                    inventory=inventory,
                    state_path=state_path,
                )
            finally:
                runtime.USER_CONFIG_DIR = old_user_dir

        self.assertEqual(ASSEMBLY_TEST_ROLE, plan.role_name)
        self.assertEqual(ASSEMBLY_TEST_SHAPE, plan.piece_id)
        self.assertEqual(1, plan.start_r)
        self.assertEqual(0, plan.start_c)
        self.assertEqual("inv_h2", plan.inventory_drive["uid"])
        self.assertEqual("plan_h2", plan.equipped_drive["uid"])

    def test_iter_blueprint_piece_placements_finds_multiple_shapes(self):
        blueprint = [
            ["XX", "XX", "XX", "XX", "XX"],
            ["H_2", "H_2", "V_2", "0", "0"],
            ["0", "0", "V_2", "0", "0"],
            ["0", "0", "0", "0", "0"],
            ["0", "0", "0", "0", "0"],
        ]
        orchestrator = NTEPipelineOrchestrator(config_dir="config")
        pieces = iter_blueprint_piece_placements(blueprint, orchestrator.shapes_db)
        self.assertEqual(2, len(pieces))
        self.assertEqual(
            [("H_2", 1, 0), ("V_2", 1, 2)],
            [(piece.piece_id, piece.start_r, piece.start_c) for piece in pieces],
        )

    def test_plan_full_assembly_reads_all_jiuyuan_pieces(self):
        blueprint = [
            ["XX", "XX", "XX", "XX", "XX"],
            ["H_2", "H_2", "V_2", "0", "0"],
            ["0", "0", "V_2", "0", "0"],
            ["0", "0", "0", "0", "0"],
            ["0", "0", "0", "0", "0"],
        ]
        equipped_state = {
            ASSEMBLY_TEST_ROLE: {
                "blueprint_layout": blueprint,
                "equipped_drives": [],
            }
        }

        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / "equipped_state.json"
            state_path.write_text(json.dumps(equipped_state, ensure_ascii=False), encoding="utf-8")
            old_user_dir = runtime.USER_CONFIG_DIR
            runtime.USER_CONFIG_DIR = Path(tmp)
            try:
                plan = plan_full_assembly(
                    role_name=ASSEMBLY_TEST_ROLE,
                    config_dir="config",
                    state_path=state_path,
                )
            finally:
                runtime.USER_CONFIG_DIR = old_user_dir

        self.assertEqual(ASSEMBLY_TEST_ROLE, plan.blueprint_role)
        self.assertEqual(2, len(plan.pieces))
        self.assertEqual("H_2", plan.pieces[0].piece_id)
        self.assertEqual("V_2", plan.pieces[1].piece_id)

    def test_find_piece_anchor_does_not_index_out_of_range_near_edge(self):
        blueprint = [
            ["XX", "XX", "XX", "XX", "XX"],
            ["H_4", "H_4", "H_4", "H_4", "0"],
            ["0", "0", "0", "0", "0"],
            ["0", "0", "0", "0", "0"],
            ["0", "0", "0", "0", "0"],
        ]
        orchestrator = NTEPipelineOrchestrator(config_dir="config")
        pieces = iter_blueprint_piece_placements(blueprint, orchestrator.shapes_db)
        self.assertEqual(1, len(pieces))
        self.assertEqual((1, 0), (pieces[0].start_r, pieces[0].start_c))

    def test_iter_blueprint_supports_jagged_rows(self):
        blueprint = [
            ["XX", "XX"],
            ["H_2", "H_2", "0"],
        ]
        orchestrator = NTEPipelineOrchestrator(config_dir="config")
        pieces = iter_blueprint_piece_placements(
            blueprint,
            orchestrator.shapes_db,
            expected_rows=5,
            expected_cols=5,
        )
        self.assertEqual(1, len(pieces))
        self.assertEqual("H_2", pieces[0].piece_id)

    def test_resolve_numeric_blueprint_cell_from_equipped_drives(self):
        blueprint = [[1, -1]]
        equipped_drives = [{"uid": "d1", "shape_id": "H_2"}]
        orchestrator = NTEPipelineOrchestrator(config_dir="config")
        pieces = iter_blueprint_piece_placements(
            blueprint,
            orchestrator.shapes_db,
            equipped_drives=equipped_drives,
            expected_rows=5,
            expected_cols=5,
        )
        self.assertEqual(1, len(pieces))
        self.assertEqual("H_2", pieces[0].piece_id)


if __name__ == "__main__":
    unittest.main()
