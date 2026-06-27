# 渲染配装结果、评分和属性汇总。
"""MainWindow methods for allocation."""

from __future__ import annotations

import json
import re

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QDialog, QDialogButtonBox, QFrame, QGroupBox, QHBoxLayout, QLabel, QPushButton, QScrollArea, QSizePolicy, QVBoxLayout, QWidget

from src.app import runtime
from src.app.constants import ALLOCATION_TOTAL_SCORE_AREA
from src.app.theme import GRADE_BGS, GRADE_COLORS, STYLE
from src.ui.puzzle_board import PuzzleBoardWidget, get_shape_pixmap as _get_shape_pixmap

from src.ui.main_window_method_install import install_methods as _install_main_window_methods

__all__ = ['_section_label', '_render_results', '_calc_grade', '_show_plan_diff_dialog', '_build_plan_diff_dialog', '_diff_item_card', '_diff_item_score_info', '_plan_diff_text', '_stat_w', '_stat_c', '_weighted_score', '_quality_coef', '_canonical_stat_name', '_stat_number_value', '_item_value', '_add_stat_total', '_fallback_tape_main_value', '_extra_shape_area', '_equipment_bonus_rows', '_format_bonus_value', '_bonus_summary_widget', '_role_stat_priority_stats', '_sort_bonus_aligned_rows', '_role_bonus_summary_panel', '_aligned_bonus_comparison_rows', '_has_bonus_delta', '_bonus_row_widget', '_bonus_placeholder_row_widget', '_bonus_spacer_row', '_bonus_comparison_column', '_bonus_delta_row_widget', '_bonus_delta_column', '_bonus_comparison_widget', '_show_bonus_summary_dialog', '_show_bonus_comparison_dialog', '_score_drive_dict', '_score_tape_dict', '_equip_card']


def install_methods(app_module, window_cls):
    """Install this feature's extracted MainWindow methods."""
    _install_main_window_methods(app_module, window_cls, __all__, globals())


def _section_label(self,text):
    label=QLabel(text)
    label.setStyleSheet("font-size:14px;font-weight:700;color:#c9d1d9;border:none;background:transparent;padding:2px 0")
    return label

def _render_results(self,plan):
    if not plan: return
    self.result_card.setVisible(True)
    while self.result_content_layout.count():
        it=self.result_content_layout.takeAt(0)
        if it.widget(): it.widget().deleteLater()
    mode_labels={"role_priority":"角色优先","drive_priority":"驱动优先","global_optimal":"全局最优","update_mode":"增量更新"}
    mode_name=mode_labels.get(getattr(self,'_pending_strat',''),'')
    plan_diffs=getattr(self,"allocation_plan_diff",{}) or {}
    for role,p in plan.items():
        if not p or not p.get("valid"):
            self.result_content_layout.addWidget(QLabel(f"❌ {role}: 无有效配装方案")); continue
        role_diff=plan_diffs.get(role,{}) or {}
        added_uids=set(role_diff.get("added_uids",set()) or set())
        total_score=p.get('score',0); total_grade=self._calc_grade(total_score,ALLOCATION_TOTAL_SCORE_AREA)
        gc=GRADE_COLORS.get(total_grade,"#58a6ff"); gbg=GRADE_BGS.get(total_grade,f"{gc}15")

        grp=QGroupBox(""); grp.setStyleSheet("QGroupBox{background:#0d1117;border:1px solid #30363d;border-radius:10px;margin-top:12px;padding:18px}")
        gl=QVBoxLayout(grp); gl.setSpacing(10)
        # Role header: name + score + grade side by side, compact
        role_hdr=QHBoxLayout(); role_hdr.setSpacing(8)
        # Role name with different color from stat blocks - use teal/cyan tone
        rnl=QLabel(role)
        rnl.setStyleSheet("font-size:15px;font-weight:800;color:#4dd0e1;border:1px solid #4dd0e1;border-radius:7px;padding:4px 14px;background:#4dd0e122")
        role_hdr.addWidget(rnl)
        if role_diff.get("changed"):
            diff_btn=QPushButton("变动")
            diff_btn.setFixedSize(76,32)
            diff_btn.setStyleSheet("QPushButton{background:#1f6feb;color:#ffffff;border:1px solid #58a6ff;border-radius:6px;font-size:13px;font-weight:700;padding:0;min-width:76px;min-height:32px}QPushButton:hover{background:#388bfd}")
            diff_btn.clicked.connect(lambda _checked=False,rn=role,d=role_diff: self._show_plan_diff_dialog(rn,d))
            role_hdr.addWidget(diff_btn)
        if mode_name:
            ml=QLabel(mode_name); ml.setStyleSheet("font-size:12px;color:#8b949e;border:1px solid #30363d;border-radius:5px;padding:3px 8px")
            role_hdr.addWidget(ml)
        role_hdr.addStretch()
        # Score badge (separate)
        sf=QFrame()
        sf.setStyleSheet(f"QFrame{{background:{gbg};border:1px solid {gc};border-radius:7px;padding:4px 12px}}")
        slb=QHBoxLayout(sf); slb.setSpacing(6); slb.setContentsMargins(4,0,4,0)
        sv=QLabel(f"{total_score:.1f}"); sv.setStyleSheet(f"font-size:15px;font-weight:800;color:{gc};border:none")
        slb.addWidget(QLabel("评分")); slb.addWidget(sv)
        role_hdr.addWidget(sf)
        # Grade badge (separate)
        gf=QFrame()
        gf.setStyleSheet(f"QFrame{{background:{gbg};border:1px solid {gc};border-radius:7px;padding:4px 12px}}")
        glb=QHBoxLayout(gf); glb.setSpacing(6); glb.setContentsMargins(4,0,4,0)
        gv=QLabel(total_grade); gv.setStyleSheet(f"font-size:15px;font-weight:800;color:{gc};border:none")
        glb.addWidget(QLabel("评级")); glb.addWidget(gv)
        role_hdr.addWidget(gf)
        gl.addLayout(role_hdr); gl.addSpacing(6)

        board=p.get("blueprint",{}).get("board",[])
        wts=self.roles_db.get(role,{}).get("weights",{})

        tape=p.get("assigned_tape")
        drives=p.get("assigned_set_drives",[])+p.get("assigned_extra_drives",[])
        if board:
            gl.addWidget(self._section_label("拼图图纸:"))
            bp_row=QHBoxLayout(); bp_row.setSpacing(44)
            bp_row.addWidget(PuzzleBoardWidget(board),0,Qt.AlignTop)
            bp_row.addWidget(self._role_bonus_summary_panel(role,tape,drives,compare_with_saved=bool(role_diff.get("changed")),priority_stats=self._role_stat_priority_stats(role)),1,Qt.AlignTop)
            gl.addLayout(bp_row); gl.addSpacing(8)

        if tape:
            t_score=tape.role_scores.get(role,0) if hasattr(tape,'role_scores') else 0
            t_grade=self._calc_grade(t_score,15)
            gl.addWidget(self._section_label("卡带:"))
            gl.addWidget(self._equip_card(tape.set_name,tape.main_stats,tape.sub_stats,None,tape.uid,wts,(t_score,t_grade),tape.quality,is_new=tape.uid in added_uids))

        if drives:
            gl.addWidget(self._section_label(f"驱动 ({len(drives)}个):"))
            for d in drives:
                score=d.role_scores.get(role,0) if hasattr(d,'role_scores') else 0
                grade=self._calc_grade(score,d.area)
                mvp_tag=f" 👑第{d.pick_order}顺位" if getattr(d,'is_mvp',False) else ""
                gl.addWidget(self._equip_card(d.shape_id,"",d.sub_stats,d.shape_id,d.uid+mvp_tag,wts,(score,grade),d.quality,is_new=d.uid in added_uids))
        self.result_content_layout.addWidget(grp)
    self.result_content_layout.addStretch()

def _calc_grade(self, score, area):
    max_score = area * 10.0
    if max_score == 0: return "D"
    ratio = score / max_score
    if ratio >= 0.8: return "ACE"
    elif ratio >= 0.7: return "SSS"
    elif ratio >= 0.6: return "SS"
    elif ratio >= 0.5: return "S"
    elif ratio >= 0.4: return "A"
    elif ratio >= 0.3: return "B"
    elif ratio >= 0.2: return "C"
    return "D"

def _plan_diff_text(self, role_name, diff):
    removed=diff.get("removed",[]) or []
    added=diff.get("added",[]) or []
    if not removed and not added:
        return "本次配装与已保存方案没有装备变动。"
    lines=[f"{role_name} 配装变动："]
    if removed:
        lines.append("\n卸下：")
        lines.extend(f"- {item.get('display_name') or item.get('uid')}" for item in removed)
    if added:
        lines.append("\n换上：")
        lines.extend(f"+ {item.get('display_name') or item.get('uid')}" for item in added)
    return "\n".join(lines)

def _diff_item_score_info(self, item):
    if "score" not in item:
        return None
    score=float(item.get("score",0.0) or 0.0)
    grade=item.get("grade")
    if not grade:
        area=int(item.get("score_area") or item.get("area") or (15 if item.get("type")=="tape" else 0) or 0)
        grade=self._calc_grade(score,area) if area else "D"
    return score,str(grade)

def _diff_value(item, key, default=None):
    if isinstance(item, dict):
        return item.get(key, default)
    return getattr(item, key, default)

def _diff_item_type(item):
    explicit=_diff_value(item,"type") or _diff_value(item,"item_type")
    if explicit:
        return str(explicit)
    if _diff_value(item,"shape_id")=="TAPE_15":
        return "tape"
    main_stats=_diff_value(item,"main_stats")
    return "tape" if isinstance(main_stats,str) and main_stats else "drive"

def _diff_grade(self, score, area):
    calc=getattr(self, "_calc_grade", None)
    if calc:
        return calc(score, area)
    return _calc_grade(self, score, area)

def _diff_snapshot_from_source(self, role_name, source):
    uid=str(_diff_value(source,"uid","") or "")
    if not uid:
        return {}
    item_type=_diff_item_type(source)
    sub_stats=_diff_value(source,"sub_stats",{}) or {}
    quality=_diff_value(source,"quality","Gold")
    area=int(_diff_value(source,"score_area") or _diff_value(source,"area") or (15 if item_type=="tape" else 0) or 0)
    role_scores=_diff_value(source,"role_scores",{}) or {}
    score=_diff_value(source,"score")
    if score is None and isinstance(role_scores,dict):
        score=role_scores.get(role_name)
    score_value=None if score is None else round(float(score or 0.0),2)
    grade=_diff_value(source,"grade")
    if grade is None and score_value is not None and area:
        grade=_diff_grade(self, score_value, area)

    snapshot={
        "uid": uid,
        "type": item_type,
        "display_name": str(_diff_value(source,"display_name","") or uid),
        "sub_stats": sub_stats,
        "quality": quality,
    }
    if item_type=="tape":
        snapshot["set_name"]=_diff_value(source,"set_name","") or "卡带"
        snapshot["main_stats"]=_diff_value(source,"main_stats","")
        snapshot["shape_id"]="TAPE_15"
    else:
        snapshot["shape_id"]=_diff_value(source,"shape_id","") or ""
    if area:
        snapshot["area"]=area
        snapshot["score_area"]=area
    if score_value is not None:
        snapshot["score"]=score_value
    if grade is not None:
        snapshot["grade"]=str(grade)
    return snapshot

def _merge_diff_item(base, source):
    merged=dict(base or {})
    for key,value in (source or {}).items():
        if key not in merged or merged[key] in (None,"",{},[]):
            merged[key]=value
    return merged

def _diff_saved_sources(self, role_name):
    role_data=(getattr(self,"equipped_state",{}) or {}).get(role_name,{})
    if not isinstance(role_data,dict):
        return []
    items=[]
    tape=role_data.get("equipped_tape")
    if isinstance(tape,dict):
        items.append(tape)
    items.extend([item for item in role_data.get("equipped_drives",[]) or [] if isinstance(item,dict)])
    return items

def _diff_plan_sources(self, role_name):
    plan=(getattr(self,"final_plan",{}) or {}).get(role_name,{})
    if not isinstance(plan,dict):
        return []
    return (
        ([plan.get("assigned_tape")] if plan.get("assigned_tape") else [])
        + list(plan.get("assigned_set_drives",[]) or [])
        + list(plan.get("assigned_extra_drives",[]) or [])
    )

def _diff_inventory_sources(self):
    output_file=getattr(runtime,"OUTPUT_FILE",None)
    if not output_file:
        return {}
    path_key=str(output_file)
    cached=getattr(self,"_diff_inventory_index_cache",None)
    if cached and cached[0]==path_key:
        return cached[1]
    index={}
    try:
        data=json.loads(output_file.read_text(encoding="utf-8"))
    except Exception:
        data=[]
    if isinstance(data,list):
        for item in data:
            if isinstance(item,dict) and item.get("uid"):
                index[str(item["uid"])]=item
    setattr(self,"_diff_inventory_index_cache",(path_key,index))
    return index

def _parse_diff_display_name(item):
    display=str(item.get("display_name") or "")
    if not display or "-" not in display:
        return {}
    shape_id, raw_stats=display.split("-",1)
    parsed={"shape_id":shape_id.strip(),"type":"drive"}
    stats={}
    for part in raw_stats.split("|"):
        part=part.strip()
        if "_" not in part:
            continue
        name,value=part.rsplit("_",1)
        try:
            stats[name.strip()]=float(str(value).replace("%","").strip())
        except Exception:
            continue
    if stats:
        parsed["sub_stats"]=stats
    return parsed

def _hydrate_diff_item(self, role_name, item):
    hydrated=dict(item or {})
    uid=str(hydrated.get("uid","") or "")
    if uid:
        for source in _diff_saved_sources(self, role_name):
            if str(source.get("uid",""))==uid:
                hydrated=_merge_diff_item(hydrated,_diff_snapshot_from_source(self,role_name,source))
                break
        if not hydrated.get("shape_id") or not hydrated.get("sub_stats") or "score" not in hydrated:
            for source in _diff_plan_sources(self, role_name):
                if str(_diff_value(source,"uid",""))==uid:
                    hydrated=_merge_diff_item(hydrated,_diff_snapshot_from_source(self,role_name,source))
                    break
        if not hydrated.get("shape_id") or not hydrated.get("sub_stats") or "score" not in hydrated:
            source=_diff_inventory_sources(self).get(uid)
            if source:
                hydrated=_merge_diff_item(hydrated,_diff_snapshot_from_source(self,role_name,source))
    if not hydrated.get("shape_id") or not hydrated.get("sub_stats"):
        hydrated=_merge_diff_item(hydrated,_parse_diff_display_name(hydrated))
    item_type=hydrated.get("type") or hydrated.get("item_type")
    if item_type:
        hydrated["type"]=item_type
    elif hydrated.get("shape_id")=="TAPE_15":
        hydrated["type"]="tape"
    else:
        hydrated["type"]="drive"
    if "score" in hydrated and "grade" not in hydrated:
        area=int(hydrated.get("score_area") or hydrated.get("area") or (15 if hydrated.get("type")=="tape" else 0) or 0)
        if area:
            hydrated["grade"]=_diff_grade(self,float(hydrated.get("score") or 0.0),area)
            hydrated["score_area"]=area
    return hydrated

def _diff_item_card(self, role_name, item, is_new=False):
    item=_hydrate_diff_item(self, role_name, item)
    weights=self.roles_db.get(role_name,{}).get("weights",{})
    score_info=getattr(self, "_diff_item_score_info", None) or (lambda diff_item: _diff_item_score_info(self, diff_item))
    item_type=item.get("type","drive")
    if item_type=="tape":
        label=item.get("set_name") or "卡带"
        main_stat=item.get("main_stats","")
        shape_id=None
    else:
        label=item.get("shape_id") or item.get("display_name") or item.get("uid","")
        main_stat=""
        shape_id=item.get("shape_id") or ""
    return self._equip_card(
        label,
        main_stat,
        item.get("sub_stats",{}) or {},
        shape_id,
        item.get("uid",""),
        weights,
        score_info(item),
        item.get("quality","Gold"),
        is_new=is_new,
    )

def _split_loadout_sources(sources):
    tape=None
    drives=[]
    for item in sources or []:
        if not item:
            continue
        item_type=str(item.get("type") if isinstance(item,dict) else getattr(item,"type","") or "")
        if item_type=="tape" or (isinstance(item,dict) and item.get("main_stats") and not item.get("shape_id")):
            tape=item
        else:
            drives.append(item)
    return tape,drives

def _build_plan_diff_dialog(self, role_name, diff):
    dlg=QDialog(self if isinstance(self, QWidget) else None)
    dlg.setWindowTitle(f"{role_name} - 配装变动")
    dlg.setMinimumSize(820,560)
    dlg.setStyleSheet(STYLE)
    layout=QVBoxLayout(dlg); layout.setContentsMargins(14,14,14,14); layout.setSpacing(10)
    scroll=QScrollArea(); scroll.setWidgetResizable(True)
    body=QWidget(); body_layout=QVBoxLayout(body); body_layout.setContentsMargins(0,0,0,0); body_layout.setSpacing(10)
    section_label=getattr(self, "_section_label", None) or (lambda text: _section_label(self, text))
    diff_item_card=getattr(self, "_diff_item_card", None) or (lambda role, item, is_new=False: _diff_item_card(self, role, item, is_new))

    removed=diff.get("removed",[]) or []
    added=diff.get("added",[]) or []

    if not removed and not added:
        body_layout.addWidget(QLabel("本次配装与已保存方案没有装备变动。"))
    else:
        # Separate tapes and drives in both removed and added
        removed_tape=[it for it in removed if it.get("type")=="tape"]
        removed_drives=[it for it in removed if it.get("type")!="tape"]
        added_tape=[it for it in added if it.get("type")=="tape"]
        added_drives=[it for it in added if it.get("type")!="tape"]

        # Pair drives by shape family: exact shape_id match first, then same-type+area match.
        _SHAPE_FAMILY = {
            "H_2": "I_2", "V_2": "I_2",
            "H_3": "I_3", "V_3": "I_3",
            "H_4": "I_4", "V_4": "I_4",
            "L_3_TL": "L_3", "L_3_TR": "L_3", "L_3_BL": "L_3", "L_3_BR": "L_3",
            "Trap_4_H": "Trap_4", "Trap_4_V": "Trap_4",
        }
        def _shape_family(sid: str) -> str:
            """Return a family key for same-shape-class matching. Unknown shapes map to themselves."""
            return _SHAPE_FAMILY.get(sid, sid)

        def _match_pairs(old_list, new_list):
            """Two-tier pairing: exact shape_id first, then same-type+area family.
            Leftover items with no counterpart become unmatched."""
            # --- first pass: exact shape_id ---
            old_by_shape: dict[str, list] = {}
            for item in old_list:
                sid = str(item.get("shape_id", "") or "")
                old_by_shape.setdefault(sid, []).append(item)
            new_by_shape: dict[str, list] = {}
            for item in new_list:
                sid = str(item.get("shape_id", "") or "")
                new_by_shape.setdefault(sid, []).append(item)

            pairs = []
            all_exact_shapes = set(old_by_shape) | set(new_by_shape)
            for sid in sorted(all_exact_shapes):
                old_items = old_by_shape.get(sid, [])
                new_items = new_by_shape.get(sid, [])
                n = min(len(old_items), len(new_items))
                for i in range(n):
                    pairs.append((old_items[i], new_items[i]))
                # replace with leftovers
                old_by_shape[sid] = old_items[n:]
                new_by_shape[sid] = new_items[n:]

            # --- second pass: match leftovers by shape family ---
            # collect leftover items grouped by family
            old_left: list = []
            for items in old_by_shape.values():
                old_left.extend(items)
            new_left: list = []
            for items in new_by_shape.values():
                new_left.extend(items)

            old_by_family: dict[str, list] = {}
            for item in old_left:
                fam = _shape_family(str(item.get("shape_id", "") or ""))
                old_by_family.setdefault(fam, []).append(item)
            new_by_family: dict[str, list] = {}
            for item in new_left:
                fam = _shape_family(str(item.get("shape_id", "") or ""))
                new_by_family.setdefault(fam, []).append(item)

            unmatched_old = []
            unmatched_new = []

            all_families = set(old_by_family) | set(new_by_family)
            for fam in sorted(all_families):
                old_items = old_by_family.get(fam, [])
                new_items = new_by_family.get(fam, [])
                n = min(len(old_items), len(new_items))
                for i in range(n):
                    pairs.append((old_items[i], new_items[i]))
                unmatched_old.extend(old_items[n:])
                unmatched_new.extend(new_items[n:])

            return pairs, unmatched_old, unmatched_new

        pair_index=0

        # Tape comparison
        if removed_tape or added_tape:
            pair_index+=1
            body_layout.addWidget(section_label(f"变动 {pair_index}：卡带"))
            pair_frame=QFrame()
            pair_frame.setStyleSheet("QFrame{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:8px 10px}")
            pair_layout=QVBoxLayout(pair_frame); pair_layout.setSpacing(6); pair_layout.setContentsMargins(8,6,8,6)

            # Old tape
            old_lbl=QLabel("← 卸下（旧）")
            old_lbl.setStyleSheet("font-size:11px;font-weight:700;color:#f85149;border:none;background:transparent;padding:2px 4px")
            pair_layout.addWidget(old_lbl)
            if removed_tape:
                pair_layout.addWidget(diff_item_card(role_name,removed_tape[0],is_new=False))
            else:
                pair_layout.addWidget(QLabel("  （无需卸下）"))

            # Arrow
            arrow=QLabel("  ↓")
            arrow.setStyleSheet("font-size:18px;font-weight:700;color:#58a6ff;border:none;background:transparent;padding:0 0 0 12px")
            pair_layout.addWidget(arrow)

            # New tape
            new_lbl=QLabel("→ 换上（新）")
            new_lbl.setStyleSheet("font-size:11px;font-weight:700;color:#56d364;border:none;background:transparent;padding:2px 4px")
            pair_layout.addWidget(new_lbl)
            if added_tape:
                pair_layout.addWidget(diff_item_card(role_name,added_tape[0],is_new=True))
            else:
                pair_layout.addWidget(QLabel("  （无需换上）"))

            body_layout.addWidget(pair_frame)

        # Drive comparisons
        drive_pairs,unmatched_old,unmatched_new=_match_pairs(removed_drives,added_drives)

        for old_d,new_d in drive_pairs:
            pair_index+=1
            old_sid=old_d.get("shape_id","未知驱动")
            new_sid=new_d.get("shape_id","未知驱动")
            title=f"变动 {pair_index}：{old_sid} → {new_sid}" if old_sid!=new_sid else f"变动 {pair_index}：{old_sid}"
            body_layout.addWidget(section_label(title))

            pair_frame=QFrame()
            pair_frame.setStyleSheet("QFrame{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:8px 10px}")
            pair_layout=QVBoxLayout(pair_frame); pair_layout.setSpacing(6); pair_layout.setContentsMargins(8,6,8,6)

            old_lbl=QLabel("← 卸下（旧）")
            old_lbl.setStyleSheet("font-size:11px;font-weight:700;color:#f85149;border:none;background:transparent;padding:2px 4px")
            pair_layout.addWidget(old_lbl)
            pair_layout.addWidget(diff_item_card(role_name,old_d,is_new=False))

            arrow=QLabel("  ↓")
            arrow.setStyleSheet("font-size:18px;font-weight:700;color:#58a6ff;border:none;background:transparent;padding:0 0 0 12px")
            pair_layout.addWidget(arrow)

            new_lbl=QLabel("→ 换上（新）")
            new_lbl.setStyleSheet("font-size:11px;font-weight:700;color:#56d364;border:none;background:transparent;padding:2px 4px")
            pair_layout.addWidget(new_lbl)
            pair_layout.addWidget(diff_item_card(role_name,new_d,is_new=True))

            body_layout.addWidget(pair_frame)

        # Unmatched old drives (removed only, no replacement)
        for old_d in unmatched_old:
            pair_index+=1
            body_layout.addWidget(section_label(f"变动 {pair_index}：卸下 {old_d.get('shape_id','未知驱动')}"))
            body_layout.addWidget(diff_item_card(role_name,old_d,is_new=False))

        # Unmatched new drives (added only, no old counterpart)
        for new_d in unmatched_new:
            pair_index+=1
            body_layout.addWidget(section_label(f"变动 {pair_index}：新增 {new_d.get('shape_id','未知驱动')}"))
            body_layout.addWidget(diff_item_card(role_name,new_d,is_new=True))

    body_layout.addStretch()
    scroll.setWidget(body)
    layout.addWidget(scroll,1)
    buttons=QDialogButtonBox(QDialogButtonBox.Close)
    buttons.rejected.connect(dlg.reject)
    layout.addWidget(buttons)
    return dlg

def _show_plan_diff_dialog(self, role_name, diff):
    self._build_plan_diff_dialog(role_name,diff).exec()

def _stat_w(self,sn,wts):
    if not wts: return 0.0
    if sn in wts: return wts[sn]
    for k,v in wts.items():
        if k.replace("%","")==sn.replace("%","") or sn.replace("%","")==k.replace("%",""): return v
        if k in sn or sn in k: return v
    return 0.0

def _stat_c(self,w):
    w=max(0.0,min(1.0,w))
    if w<0.3: return "#8b949e"
    if w<0.5: return "#58a6ff"
    if w<0.7: return "#56d364"
    if w<0.85: return "#d2991d"
    return "#f0883e"

def _weighted_score(self,sub_stats,wts):
    if not sub_stats: return 0
    total=0.0
    for sn,sv in sub_stats.items():
        sw=self._stat_w(sn,wts)
        total+=float(sv)*sw
    return total

def _quality_coef(self, quality):
    return {"Gold":1.0,"Purple":0.8,"Blue":0.6}.get(str(quality or "Gold"),1.0)

def _canonical_stat_name(self, stat):
    stat=str(stat or "").strip()
    if not stat:
        return ""
    aliases={}
    if self.scoring_engine:
        aliases=getattr(self.scoring_engine,"stat_alias_mapping",{}) or {}
    aliases.update(self.stats_config.get("stat_alias_mapping",{}) if isinstance(self.stats_config,dict) else {})
    return aliases.get(stat,stat)

def _stat_number_value(self, value):
    try:
        return float(str(value).replace("%","").strip())
    except Exception:
        return 0.0

def _item_value(self, item, key, default=None):
    if isinstance(item,dict):
        return item.get(key,default)
    return getattr(item,key,default)

def _add_stat_total(self, totals, stat, value):
    stat=self._canonical_stat_name(stat)
    value=self._stat_number_value(value)
    if not stat or value==0:
        return
    totals[stat]=round(totals.get(stat,0.0)+value,4)

def _fallback_tape_main_value(self, main_stat, quality):
    configured=(self.stats_config or {}).get("tape_main_stat_values",{})
    main_stat=str(main_stat or "").strip()
    canonical=self._canonical_stat_name(main_stat)
    if main_stat in configured:
        return self._stat_number_value(configured[main_stat])*self._quality_coef(quality)
    if canonical in configured:
        return self._stat_number_value(configured[canonical])*self._quality_coef(quality)
    if canonical in {"暴击伤害%"}:
        return 60.0*self._quality_coef(quality)
    if canonical in {"暴击率%"}:
        return 30.0*self._quality_coef(quality)
    if canonical in {"攻击力%","防御力%","生命值%"}:
        return 37.5*self._quality_coef(quality)
    if canonical in {"环合强度","倾陷强度"}:
        return 180.0*self._quality_coef(quality)
    if "治疗加成" in canonical:
        return 34.5*self._quality_coef(quality)
    if "伤害增强" in canonical:
        return 37.5*self._quality_coef(quality)
    return 0.0

def _extra_shape_area(self, role_name):
    label=str(self.roles_db.get(role_name,{}).get("extra_shape_label",""))
    m=re.search(r"(\d+)",label)
    return int(m.group(1)) if m else None

def _equipment_bonus_rows(self, role_name, tape, drives):
    totals={}
    if tape:
        main_stat=self._item_value(tape,"main_stats","")
        main_value=self._item_value(tape,"main_value",None)
        if main_value is None:
            main_value=self._fallback_tape_main_value(main_stat,self._item_value(tape,"quality","Gold"))
        self._add_stat_total(totals,main_stat,main_value)
        for stat,value in (self._item_value(tape,"sub_stats",{}) or {}).items():
            self._add_stat_total(totals,stat,value)
    drives=list(drives or [])
    for drive in drives:
        for stat,value in (self._item_value(drive,"sub_stats",{}) or {}).items():
            self._add_stat_total(totals,stat,value)
    role_data=self.roles_db.get(role_name,{})
    extra_buffs=role_data.get("extra_shape_buffs",{}) or {}
    if isinstance(extra_buffs,dict) and len(extra_buffs)>1:
        first_key=next(iter(extra_buffs))
        extra_buffs={first_key:extra_buffs[first_key]}
    target_area=self._extra_shape_area(role_name)
    matched_count=0
    if target_area:
        for drive in drives:
            area=self._item_value(drive,"area",None)
            if area is None:
                area=self._shape_areas.get(self._item_value(drive,"shape_id",""),0)
            if int(area or 0)==target_area:
                matched_count+=1
    for stat,value in extra_buffs.items():
        self._add_stat_total(totals,stat,self._stat_number_value(value)*matched_count)
    rows=sorted(totals.items(),key=lambda kv: kv[1],reverse=True)
    return [(stat,value) for stat,value in rows if value]

def _format_bonus_value(self, stat, value):
    suffix="%" if "%" in stat or "伤害增强" in stat or "治疗加成" in stat else ""
    if suffix:
        return f"+{value:.2f}%"
    return f"+{value:.0f}" if abs(value-round(value))<0.01 else f"+{value:.2f}"

def _is_crit_rate_stat(stat):
    normalized=str(stat or "").replace("%","").strip()
    return normalized in {"暴击率","暴击率%"}

def _stats_match(stat, stat_key):
    left=str(stat or "").replace("%","").strip()
    right=str(stat_key or "").replace("%","").strip()
    if not left or not right:
        return False
    return left == right or left in right or right in left

def _is_highlighted_bonus_stat(stat, priority_stats=None):
    if _is_crit_rate_stat(stat):
        return True
    for key in priority_stats or []:
        if _stats_match(stat, key):
            return True
    return False

def _bonus_stat_label_style(stat, priority_stats=None):
    color="#d2991d" if _is_highlighted_bonus_stat(stat, priority_stats) else "#c9d1d9"
    return f"font-size:10px;font-weight:700;color:{color};border:none;background:transparent"

def _role_stat_priority_stats(self, role_name):
    configs=getattr(self,"_pending_crit_priority_modes",None) or {}
    if not configs and hasattr(self,"role_selector"):
        try:
            configs=self.role_selector.get_crit_priority_modes()
        except Exception:
            configs={}
    cfg=configs.get(role_name) or {}
    if not isinstance(cfg,dict):
        return []
    return [str(stat) for stat in cfg.get("stats", []) if stat]

def _sort_bonus_aligned_rows(self, aligned, priority_stats=None, prioritize_changed_only=False):
    priority_stats=list(priority_stats or [])

    def priority_index(stat, item):
        if prioritize_changed_only and not self._has_bonus_delta(item):
            return None
        for idx,key in enumerate(priority_stats):
            if _stats_match(stat,key):
                return idx
        if _is_crit_rate_stat(stat):
            return len(priority_stats)
        return None

    def sort_key(item):
        stat=item.get("stat","")
        idx=priority_index(stat, item)
        if idx is not None:
            return (0,idx)
        max_val=max(float(item.get("old") or 0.0),float(item.get("new") or 0.0))
        return (1,-max_val)

    return sorted(aligned or [], key=sort_key)

def _has_bonus_delta(self, item):
    delta=float(item.get("delta") or 0.0)
    if abs(delta) < 0.0001:
        return False
    old_val=item.get("old")
    new_val=item.get("new")
    if old_val is not None and new_val is not None and old_val==new_val:
        return False
    return True

def _aligned_bonus_comparison_rows(self, old_rows, new_rows, limit=None, changes_only=False, priority_stats=None):
    old_map=dict(old_rows or [])
    new_map=dict(new_rows or [])
    stats=set(old_map) | set(new_map)
    aligned=[]
    for stat in stats:
        old_val=old_map.get(stat)
        new_val=new_map.get(stat)
        if old_val is not None and new_val is not None:
            delta=round(new_val-old_val,4)
        elif old_val is None and new_val is not None:
            delta=round(new_val,4)
        elif new_val is None and old_val is not None:
            delta=round(-old_val,4)
        else:
            delta=0.0
        aligned.append({"stat": stat, "old": old_val, "new": new_val, "delta": delta})
    if changes_only:
        aligned=[item for item in aligned if self._has_bonus_delta(item)]
    aligned=self._sort_bonus_aligned_rows(aligned,priority_stats,prioritize_changed_only=changes_only)
    if limit is not None:
        aligned=aligned[:limit]
    return aligned

def _bonus_spacer_row(self):
    row=QFrame()
    row.setFixedHeight(26)
    row.setStyleSheet("QFrame{background:transparent;border:none;}")
    return row

def _bonus_placeholder_row_widget(self, stat, text="—", priority_stats=None):
    row=QFrame()
    row.setFixedHeight(26)
    row.setMinimumWidth(130)
    row.setStyleSheet("QFrame{background:#161b22;border:1px solid #21262d;border-radius:5px;padding:2px 6px}")
    rl=QHBoxLayout(row); rl.setContentsMargins(6,1,6,1); rl.setSpacing(6)
    name=QLabel(stat); name.setWordWrap(True); name.setStyleSheet(_bonus_stat_label_style(stat,priority_stats))
    val=QLabel(text); val.setAlignment(Qt.AlignRight|Qt.AlignVCenter)
    val.setStyleSheet("font-size:10px;font-weight:700;color:#6e7681;border:none;background:transparent")
    rl.addWidget(name,1); rl.addWidget(val)
    return row

def _bonus_comparison_column(self, title, aligned_rows, value_key, empty_text="暂无可汇总属性", priority_stats=None):
    column=QFrame()
    column.setStyleSheet("QFrame{background:#0d1117;border:1px solid #30363d;border-radius:8px;padding:6px}")
    layout=QVBoxLayout(column); layout.setContentsMargins(7,5,7,5); layout.setSpacing(4)
    header=QLabel(title)
    header.setStyleSheet("font-size:11px;font-weight:800;color:#8b949e;border:none;background:transparent")
    layout.addWidget(header)
    if not aligned_rows:
        empty=QLabel(empty_text)
        empty.setStyleSheet("color:#6e7681;border:none;background:transparent")
        layout.addWidget(empty)
    else:
        for item in aligned_rows:
            value=item.get(value_key)
            if value is None:
                layout.addWidget(self._bonus_placeholder_row_widget(item["stat"], priority_stats=priority_stats))
            else:
                layout.addWidget(self._bonus_row_widget(item["stat"], value, priority_stats=priority_stats))
    layout.addStretch()
    return column

def _bonus_delta_row_widget(self, stat, delta, old_val, new_val, priority_stats=None):
    if not self._has_bonus_delta({"stat": stat, "delta": delta, "old": old_val, "new": new_val}):
        return self._bonus_spacer_row()
    row=QFrame()
    row.setFixedHeight(26)
    row.setMinimumWidth(130)
    row.setStyleSheet("QFrame{background:#161b22;border:1px solid #21262d;border-radius:5px;padding:2px 6px}")
    rl=QHBoxLayout(row); rl.setContentsMargins(6,1,6,1); rl.setSpacing(6)
    name=QLabel(stat); name.setWordWrap(True); name.setStyleSheet(_bonus_stat_label_style(stat,priority_stats))
    sign="+" if delta>=0 else ""
    suffix="%" if "%" in stat or "伤害增强" in stat or "治疗加成" in stat else ""
    text=f"{sign}{delta:.2f}{suffix}" if suffix else (f"{sign}{delta:.0f}" if abs(delta-round(delta))<0.01 else f"{sign}{delta:.2f}")
    color="#56d364" if delta>0 else "#f85149"
    val=QLabel(text); val.setAlignment(Qt.AlignRight|Qt.AlignVCenter)
    val.setStyleSheet(f"font-size:10px;font-weight:800;color:{color};border:none;background:transparent")
    rl.addWidget(name,1); rl.addWidget(val)
    return row

def _bonus_delta_column(self, aligned_rows, priority_stats=None):
    column=QFrame()
    column.setStyleSheet("QFrame{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:6px}")
    layout=QVBoxLayout(column); layout.setContentsMargins(7,5,7,5); layout.setSpacing(4)
    title=QLabel("变化")
    title.setStyleSheet("font-size:11px;font-weight:800;color:#8b949e;border:none;background:transparent")
    layout.addWidget(title)
    if not aligned_rows:
        empty=QLabel("无变化")
        empty.setStyleSheet("color:#6e7681;border:none;background:transparent")
        layout.addWidget(empty)
    else:
        for item in aligned_rows:
            layout.addWidget(self._bonus_delta_row_widget(item["stat"], item["delta"], item.get("old"), item.get("new"), priority_stats=priority_stats))
    layout.addStretch()
    return column

def _bonus_comparison_widget(self, role_name, old_rows, new_rows, has_old=True, compact=False, priority_stats=None):
    priority_stats=list(priority_stats or [])
    if compact:
        aligned=self._aligned_bonus_comparison_rows(old_rows,new_rows,changes_only=True,priority_stats=priority_stats)
    else:
        aligned=self._aligned_bonus_comparison_rows(old_rows,new_rows,priority_stats=priority_stats)
    old_title="旧" if compact else "旧方案"
    new_title="新" if compact else "新方案"
    old_empty="无已保存配装" if not has_old else ("暂无属性变化" if compact else "暂无可汇总属性")
    old_column=self._bonus_comparison_column(old_title,aligned,"old",old_empty,priority_stats=priority_stats)
    new_column=self._bonus_comparison_column(new_title,aligned,"new","暂无属性变化" if compact and not aligned else "暂无可汇总属性",priority_stats=priority_stats)

    container=QFrame()
    container.setStyleSheet("QFrame{background:transparent;border:none}")
    layout=QHBoxLayout(container); layout.setContentsMargins(0,0,0,0); layout.setSpacing(8)
    layout.addWidget(old_column,1)
    layout.addWidget(new_column,1)
    layout.addWidget(self._bonus_delta_column(aligned,priority_stats=priority_stats),1)
    return container

def _role_bonus_summary_panel(self, role_name, tape, drives, compare_with_saved=False, priority_stats=None):
    priority_stats=list(priority_stats if priority_stats is not None else self._role_stat_priority_stats(role_name))
    if compare_with_saved:
        saved_sources=_diff_saved_sources(self,role_name)
        old_tape,old_drives=_split_loadout_sources(saved_sources)
        if old_tape or old_drives:
            old_rows=self._equipment_bonus_rows(role_name,old_tape,old_drives)
            new_rows=self._equipment_bonus_rows(role_name,tape,drives)
            box=QFrame()
            box.setMinimumWidth(560)
            box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
            box.setStyleSheet("QFrame{background:#0d1117;border:1px solid #30363d;border-radius:8px;padding:6px}")
            layout=QVBoxLayout(box); layout.setContentsMargins(7,5,7,5); layout.setSpacing(4)
            title=QLabel("属性汇总")
            title.setStyleSheet("font-size:11px;font-weight:800;color:#8b949e;border:none;background:transparent")
            layout.addWidget(title)
            layout.addWidget(self._bonus_comparison_widget(role_name,old_rows,new_rows,has_old=True,compact=True,priority_stats=priority_stats))
            full_rows=self._aligned_bonus_comparison_rows(old_rows,new_rows,priority_stats=priority_stats)
            changed_rows=self._aligned_bonus_comparison_rows(old_rows,new_rows,changes_only=True,priority_stats=priority_stats)
            if len(full_rows)>len(changed_rows):
                more=QPushButton("•••")
                more.setObjectName("btnSm")
                more.setFixedSize(54,22)
                more.setCursor(Qt.PointingHandCursor)
                more.setStyleSheet("QPushButton{background:#161b22;color:#c9d1d9;border:1px solid #30363d;border-radius:8px;font-size:13px;font-weight:800;padding:0}QPushButton:hover{border-color:#58a6ff;color:#58a6ff}")
                more.clicked.connect(
                    lambda checked=False,role=role_name,old_r=old_rows,new_r=new_rows,stats=list(priority_stats): self._show_bonus_comparison_dialog(role,old_r,new_r,stats)
                )
                layout.addWidget(more,0,Qt.AlignCenter)
            layout.addStretch()
            return box
    return self._bonus_summary_widget(role_name,tape,drives)

def _show_bonus_comparison_dialog(self, role_name, old_rows, new_rows, priority_stats=None):
    priority_stats=list(priority_stats if priority_stats is not None else self._role_stat_priority_stats(role_name))
    dlg=QDialog(self)
    dlg.setWindowTitle(f"{role_name} 属性汇总对比")
    dlg.setMinimumSize(680,360)
    dlg.setStyleSheet(STYLE)
    layout=QVBoxLayout(dlg); layout.setContentsMargins(14,14,14,14); layout.setSpacing(8)
    layout.addWidget(self._bonus_comparison_widget(role_name,old_rows,new_rows,has_old=True,compact=False,priority_stats=priority_stats))
    buttons=QDialogButtonBox(QDialogButtonBox.Ok)
    buttons.accepted.connect(dlg.accept)
    layout.addWidget(buttons)
    dlg.exec()

def _bonus_summary_widget(self, role_name, tape, drives):
    rows=self._equipment_bonus_rows(role_name,tape,drives)
    box=QFrame()
    box.setFixedWidth(240)
    box.setStyleSheet("QFrame{background:#0d1117;border:1px solid #30363d;border-radius:8px;padding:6px}")
    layout=QVBoxLayout(box); layout.setContentsMargins(7,5,7,5); layout.setSpacing(4)
    title=QLabel("属性汇总")
    title.setStyleSheet("font-size:11px;font-weight:800;color:#8b949e;border:none;background:transparent")
    layout.addWidget(title)
    visible=rows[:4]
    if not visible:
        empty=QLabel("暂无可汇总属性")
        empty.setStyleSheet("color:#6e7681;border:none;background:transparent")
        layout.addWidget(empty)
    for stat,value in visible:
        layout.addWidget(self._bonus_row_widget(stat,value))
    if len(rows)>len(visible):
        more=QPushButton("•••")
        more.setObjectName("btnSm")
        more.setFixedSize(54,22)
        more.setCursor(Qt.PointingHandCursor)
        more.setStyleSheet("QPushButton{background:#161b22;color:#c9d1d9;border:1px solid #30363d;border-radius:8px;font-size:13px;font-weight:800;padding:0}QPushButton:hover{border-color:#58a6ff;color:#58a6ff}")
        more.clicked.connect(lambda checked=False,r=rows,role=role_name: self._show_bonus_summary_dialog(role,r))
        layout.addWidget(more,0,Qt.AlignCenter)
    layout.addStretch()
    return box

def _bonus_row_widget(self, stat, value, priority_stats=None):
    row=QFrame()
    row.setFixedHeight(26)
    row.setMinimumWidth(130)
    row.setStyleSheet("QFrame{background:#161b22;border:1px solid #21262d;border-radius:5px;padding:2px 6px}")
    rl=QHBoxLayout(row); rl.setContentsMargins(6,1,6,1); rl.setSpacing(6)
    name=QLabel(stat); name.setWordWrap(True); name.setStyleSheet(_bonus_stat_label_style(stat,priority_stats))
    val=QLabel(self._format_bonus_value(stat,value)); val.setAlignment(Qt.AlignRight|Qt.AlignVCenter)
    val.setStyleSheet("font-size:10px;font-weight:800;color:#f0f6fc;border:none;background:transparent")
    rl.addWidget(name,1); rl.addWidget(val)
    return row

def _show_bonus_summary_dialog(self, role_name, rows):
    dlg=QDialog(self)
    dlg.setWindowTitle(f"{role_name} 属性汇总")
    dlg.setMinimumSize(360,420)
    dlg.setStyleSheet(STYLE)
    layout=QVBoxLayout(dlg); layout.setContentsMargins(14,14,14,14); layout.setSpacing(8)
    for stat,value in rows:
        layout.addWidget(self._bonus_row_widget(stat,value))
    buttons=QDialogButtonBox(QDialogButtonBox.Ok)
    buttons.accepted.connect(dlg.accept)
    layout.addWidget(buttons)
    dlg.exec()

def _score_drive_dict(self, sub_stats, shape_id, weights, quality="Gold"):
    if not self.scoring_engine: return 0.0
    se=self.scoring_engine
    max_w=se._get_max_theoretical_weight(weights)
    area=self._shape_areas.get(shape_id, 3)
    actual_w=sum(se._get_flexible_weight(sn, weights) for sn in sub_stats.keys())
    if actual_w<=0 or max_w<=0: return 0.0
    quality_coef=se.quality_map.get(quality, 1.0)
    return round((10.0/max_w)*actual_w*area*quality_coef, 2)

def _score_tape_dict(self, main_stats, sub_stats, weights, quality="Gold"):
    if not self.scoring_engine: return 0.0
    se=self.scoring_engine
    max_w=se._get_max_theoretical_weight(weights)
    quality_coef=se.quality_map.get(quality, 1.0)
    main_w=se._get_flexible_weight(main_stats, weights) if main_stats else 0
    main_score=main_w*50.0*quality_coef
    sub_w=sum(se._get_flexible_weight(sn, weights) for sn in sub_stats.keys())
    sub_score=(10.0/max_w)*sub_w*10.0*quality_coef if max_w>0 else 0
    return round(main_score+sub_score, 2)

def _equip_card(self,label,main_stat,sub_stats,shape_id,uid,weights,score_info=None,quality=None,is_new=False):
    QUALITY_COLORS={"Gold":"#ffd700","Purple":"#ffe082","Blue":"#58a6ff"}
    QUALITY_LABELS={"Gold":"金","Purple":"紫","Blue":"蓝"}
    QUALITY_BGS={"Gold":"#332600","Purple":"#6f2dbd","Blue":"#0d2748"}
    w=QWidget(); w.setStyleSheet("QWidget{background:#0d1117;border:1px solid #30363d;border-radius:10px;padding:9px 13px;margin:3px 0}")
    outer=QHBoxLayout(w); outer.setSpacing(12); outer.setContentsMargins(2,2,2,2)

    # Shape image (compact)
    if shape_id:
        pm=_get_shape_pixmap(shape_id,64,quality)
        if not pm.isNull():
            img_lbl=QLabel(); img_lbl.setPixmap(pm); img_lbl.setFixedSize(68,68); img_lbl.setScaledContents(True)
            img_lbl.setStyleSheet("border:1px solid #30363d;border-radius:6px;background:#161b22"); outer.addWidget(img_lbl)

    inner=QVBoxLayout(); inner.setSpacing(5); inner.setContentsMargins(0,3,0,3)

    # Header: shape name + quality + main stat block + score|grade
    hdr=QHBoxLayout(); hdr.setSpacing(8)
    label_color = "#7ee787" if not shape_id else "#4dd0e1"
    label_bg = "#0f3d2e" if not shape_id else f"{label_color}15"
    label_border = "#238636" if not shape_id else label_color
    name_lbl = QLabel(f"<b>{label}</b>")
    name_size = 12 if shape_id else 13
    name_pad = "2px 8px" if shape_id else "3px 10px"
    name_lbl.setStyleSheet(f"font-size:{name_size}px;font-weight:800;color:{label_color};border:1px solid {label_border};border-radius:6px;padding:{name_pad};background:{label_bg}")
    hdr.addWidget(name_lbl)
    if is_new:
        new_lbl=QLabel("NEW")
        new_lbl.setStyleSheet("font-size:10px;font-weight:800;color:#58a6ff;border:1px solid #58a6ff;border-radius:5px;padding:2px 6px;background:#1f6feb22")
        hdr.addWidget(new_lbl)
    # Quality badge: only tapes show text; drive quality is represented by the icon.
    if quality and not shape_id:
        qcolor=QUALITY_COLORS.get(quality,"#8b949e"); qlabel=QUALITY_LABELS.get(quality,quality)
        qbg=QUALITY_BGS.get(quality,f"{qcolor}15")
        q_lbl=QLabel(qlabel)
        q_lbl.setStyleSheet(f"font-size:11px;font-weight:700;color:{qcolor};border:1px solid {qcolor};border-radius:5px;padding:2px 7px;background:{qbg}")
        hdr.addWidget(q_lbl)
    # Main stat as colored block (same style as sub stats)
    if main_stat:
        mw=self._stat_w(main_stat,weights); mc=self._stat_c(mw); qc=QColor(mc)
        ms_block=QLabel(main_stat); ms_block.setStyleSheet(
            f"border:1px solid {mc};background:rgba({qc.red()},{qc.green()},{qc.blue()},0.12);"
            f"border-radius:6px;padding:4px 12px;font-size:13px;color:{mc};font-weight:700"
        )
        hdr.addWidget(ms_block)
    hdr.addStretch()

    # Score | Grade side by side
    if score_info is not None:
        score,grade=score_info; gc=GRADE_COLORS.get(grade,"#58a6ff")
        sf=QFrame()
        sf.setStyleSheet(f"QFrame{{background:{gc}15;border:1px solid {gc};border-radius:6px;padding:2px 10px}}")
        sf_layout=QHBoxLayout(sf); sf_layout.setSpacing(5); sf_layout.setContentsMargins(4,1,4,1)
        sl=QLabel(f"{score:.1f}"); sl.setStyleSheet(f"font-size:13px;font-weight:800;color:{gc};border:none"); sf_layout.addWidget(sl)
        gl=QLabel(grade); gl.setStyleSheet(f"font-size:11px;font-weight:800;color:{gc};border:none"); sf_layout.addWidget(gl)
        hdr.addWidget(sf)
    uid_lbl=QLabel(f"<span style='color:#6e7681;font-size:10px;'>{uid}</span>"); hdr.addWidget(uid_lbl)
    inner.addLayout(hdr)

    # Stat blocks row
    if sub_stats:
        br=QHBoxLayout(); br.setSpacing(5)
        for sn,sv in sub_stats.items():
            sw=self._stat_w(sn,weights); color=self._stat_c(sw); qc=QColor(color)
            block=QLabel(f"{sn} <b>{sv}</b>"); block.setAlignment(Qt.AlignCenter)
            block.setStyleSheet(f"border:1px solid {color};background:rgba({qc.red()},{qc.green()},{qc.blue()},0.12);border-radius:6px;padding:5px 12px;font-size:12px;color:{color};font-weight:600")
            block.setToolTip(f"权重: {sw:.2f}"); br.addWidget(block)
        br.addStretch(); inner.addLayout(br)
    outer.addLayout(inner,1); return w
