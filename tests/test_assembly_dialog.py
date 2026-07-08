import unittest

import numpy as np

from src.scanner.assembly_dialog import dialog_text_matches


class AssemblyDialogTests(unittest.TestCase):
    def test_dialog_text_matches_equip_transfer_keywords(self):
        texts = ["提示", "该装备已镶嵌在【薄荷】角色上，是否镶嵌于此处？"]
        self.assertTrue(dialog_text_matches(texts, ["是否镶嵌于此处"]))
        self.assertTrue(dialog_text_matches(texts, ["该装备已镶嵌"]))

    def test_dialog_text_matches_requires_keyword(self):
        texts = ["提示", "驱动块详情"]
        self.assertFalse(dialog_text_matches(texts, ["是否镶嵌于此处"]))

    def test_center_dialog_crop_keeps_shape(self):
        from src.scanner.assembly_dialog import center_dialog_crop

        image = np.zeros((100, 200, 3), dtype=np.uint8)
        crop = center_dialog_crop(image)
        self.assertEqual(3, crop.ndim)
        self.assertGreater(crop.shape[0], 0)
        self.assertGreater(crop.shape[1], 0)


if __name__ == "__main__":
    unittest.main()
