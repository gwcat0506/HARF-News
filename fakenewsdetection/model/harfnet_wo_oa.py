"""
HARFNET-VER3-WO-OA
===================
Ablation: OverAlignModule 제거
- OverAlignModule (self.om) 제거
- BilinearInteraction → 2-way (auth × ver) 로 교체
- head4 입력: [F_sty, F_ver, F_bi]  (d*3)
- 제거 손실: overalign_loss, af_conditional_loss
"""

from __future__ import annotations

import argparse
import copy
import gc
import io
import os
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold, train_test_split

import harfnet as base

LOG_TAG   = "HARFNET-VER3-WO-OA"
LOG_BRAND = "HARFNET-VER3-WO-OA"


# ─────────────────────────────────────────────────────────────
# 1. 2-way BilinearInteraction (auth × ver only)
# ─────────────────────────────────────────────────────────────
class BilinearInteraction2Way(nn.Module):
    def __init__(self, d: int, r: int = base.BILINEAR_RANK):
        super().__init__()
        self.auth_low    = nn.Sequential(nn.Linear(d, r), nn.LayerNorm(r), nn.GELU())
        self.ver_low     = nn.Sequential(nn.Linear(d, r), nn.LayerNorm(r), nn.GELU())
        self.bi_proj     = nn.Sequential(nn.Linear(r, d), nn.LayerNorm(d), nn.GELU())
        self.scalar_proj = nn.Sequential(
            nn.Linear(3, 128), nn.ReLU(), nn.Linear(128, d))

    def forward(self, F_sty, F_ver, p_auth, p_fake, has_image):
        m        = has_image.unsqueeze(-1)
        hadamard = self.auth_low(F_sty) * self.ver_low(F_ver)
        F_had    = self.bi_proj(hadamard)
        scalars  = torch.stack([
            p_auth * has_image,
            p_fake * has_image,
            p_auth * p_fake * has_image,
        ], dim=-1)
        return (F_had + self.scalar_proj(scalars)) * m


# ─────────────────────────────────────────────────────────────
# 2. 모델
# ─────────────────────────────────────────────────────────────
class HARFNETver3_WO_OA(nn.Module):
    def __init__(self, roberta_name=base.ROBERTA, clip_name=base.CLIP_RN101,
                 n_heads=8, dropout=0.1):
        super().__init__()
        d = 768
        self.text_encoder = base.RobertaModel.from_pretrained(
            roberta_name, add_pooling_layer=False)
        _clip, _ = base.openai_clip.load(clip_name, device="cpu")
        self.clip_model = _clip.float()
        self.clip_model.eval()
        for p in self.clip_model.parameters():
            p.requires_grad_(False)

        self.visual    = self.clip_model.visual
        clip_edim      = self.visual.attnpool.c_proj.out_features
        self.deep_proj = nn.Linear(2048, d)
        self.cls_proj  = nn.Linear(clip_edim, d)

        self.am = base.AuthorshipModule(d, n_heads, dropout)
        self.vm = base.VeracityModule(d, n_heads, dropout)
        self.bi = BilinearInteraction2Way(d, base.BILINEAR_RANK)
        # self.om 없음

        self.head4 = nn.Sequential(
            nn.Linear(d * 3, 512),          # d*3 : F_sty, F_ver, F_bi
            nn.LayerNorm(512), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(512, 4))

    def _clip_visual_forward(self, pix, has_image):
        vis = self.visual
        B   = pix.size(0)
        m4d = has_image.view(B, 1, 1, 1)
        x   = vis.relu1(vis.bn1(vis.conv1(pix)))
        x   = vis.relu2(vis.bn2(vis.conv2(x)))
        x   = vis.relu3(vis.bn3(vis.conv3(x)))
        x   = vis.avgpool(x) * m4d
        x   = vis.layer1(x); x = vis.layer2(x); x = vis.layer3(x)
        deep     = vis.layer4(x)
        v_global = vis.attnpool(deep)
        ms       = has_image.view(B, 1, 1)
        V_pat    = self.deep_proj(deep.flatten(2).transpose(1, 2)) * ms
        V_cls    = self.cls_proj(v_global) * has_image.view(B, 1)
        return V_pat, V_cls

    def train(self, mode=True):
        super().train(mode)
        self.clip_model.eval()
        return self

    def forward(self, input_ids, attention_mask, pixel_values,
                has_image, clip_text_tokens=None):
        device    = input_ids.device
        has_image = has_image.to(device)
        B         = input_ids.size(0)

        T_tok    = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask).last_hidden_state
        txt_mask = attention_mask.bool()
        T_cls    = T_tok[:, 0]

        V_pat, V_cls = self._clip_visual_forward(pixel_values, has_image)

        F_sty, p_auth, z_auth = self.am(T_tok, txt_mask)
        F_ver, p_fake, z_ver, global_sim = self.vm(
            T_tok, txt_mask, V_pat, T_cls, V_cls, has_image)

        F_bi = self.bi(F_sty, F_ver, p_auth, p_fake, has_image)

        logits4 = torch.nan_to_num(
            self.head4(torch.cat([F_sty, F_ver, F_bi], dim=-1)),
            nan=0., posinf=20., neginf=-20.)

        zeros = torch.zeros(B, device=device, dtype=F_sty.dtype)
        return {
            "logits4":      logits4,
            "p_auth":       p_auth,
            "p_fake":       p_fake,
            "p_af":         zeros,
            "z_auth":       z_auth,
            "z_ver":        z_ver,
            "global_sim":   global_sim,
            "oa_mean":      zeros,
            "oa_std":       zeros,
            "oa_uniformity": zeros,
        }


# ─────────────────────────────────────────────────────────────
# 3. 손실 함수
# ─────────────────────────────────────────────────────────────
def total_loss_wo_oa(out, y, has_image, args):
    loss  = base.focal_loss(out["logits4"], y)
    loss += args.lambda_fake_bce * base.attribute_bce_loss(
        out["p_fake"], (y % 2 == 1), has_image)
    loss += args.lambda_auth_bce * base.attribute_bce_loss(
        out["p_auth"], (y >= 2), has_image)
    loss += args.lambda_con * base.axis_supcon_loss(
        out["z_ver"], (y % 2).long())
    # overalign_loss / af_conditional_loss 제거
    return loss


# ─────────────────────────────────────────────────────────────
# 4. 학습 / 평가
# ─────────────────────────────────────────────────────────────
def run_epoch(model, loader, device, optimizer, train,
              epoch_idx=None, args=None):
    model.train() if train else model.eval()
    tot, ys, ps, n = 0., [], [], 0
    ctx  = torch.enable_grad() if train else torch.no_grad()
    desc = f"[{LOG_TAG}] Epoch {epoch_idx}" if (train and epoch_idx) else None
    it   = base._tqdm(loader, desc=desc) if (train and desc) else loader
    with ctx:
        for batch in it:
            y   = batch["label"].to(device)
            hi  = batch["has_image"].to(device)
            out = base._fwd(model, batch, device)
            loss = total_loss_wo_oa(out, y, hi, args)
            if train and optimizer:
                optimizer.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            pred = out["logits4"].argmax(-1)
            tot += loss.item() * y.size(0)
            ys.extend(y.cpu().tolist())
            ps.extend(pred.cpu().tolist())
            n += y.size(0)
    n = max(n, 1)
    return {
        "loss":     tot / n,
        "acc4":     float(np.mean(np.array(ys) == np.array(ps))),
        "macro_f1": base.f1_score(ys, ps, average="macro", zero_division=0),
    }


def _monitor(model, loader, device) -> str:
    model.eval()
    try:
        with torch.no_grad():
            bufs  = {k: [] for k in ["lbl", "hi", "pf", "pa", "gs"]}
            found = {i: False for i in range(4)}
            for b in loader:
                out = base._fwd(model, b, device)
                lbl = b["label"]; hi = b["has_image"]
                bufs["lbl"].append(lbl);  bufs["hi"].append(hi)
                bufs["pf"].append(out["p_fake"].cpu())
                bufs["pa"].append(out["p_auth"].cpu())
                bufs["gs"].append(out["global_sim"].cpu())
                for ci in range(4):
                    if ((lbl == ci) & (hi > 0.5)).any():
                        found[ci] = True
                if all(found.values()): break
            lbl = torch.cat(bufs["lbl"]); hi = torch.cat(bufs["hi"])
            pf  = torch.cat(bufs["pf"]); pa = torch.cat(bufs["pa"])
            gs  = torch.cat(bufs["gs"])
            parts = []
            for ci, cn in enumerate(base.CLASS_NAMES):
                mask = (lbl == ci) & (hi > 0.5)
                if mask.sum() > 0:
                    parts.append(
                        f"{cn}:pf={pf[mask].mean():.2f},"
                        f"pa={pa[mask].mean():.2f},"
                        f"gs={gs[mask].mean():.3f}")
            return " | " + " / ".join(parts) if parts else ""
    except Exception:
        return ""


def train_one_run(model, tl, vl, device, args, log=None):
    opt   = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-4)
    early = base.EarlyStopping(args.early_stop_patience, args.early_stop_min_delta)
    best_f1, best_st = -1., None

    for ep in range(1, args.epochs + 1):
        tr  = run_epoch(model, tl, device, opt, True, ep, args)
        va  = run_epoch(model, vl, device, None, False, args=args)
        mon = _monitor(model, vl, device)
        line = (f"[{LOG_TAG}] Epoch {ep} "
                f"loss={tr['loss']:.4f} val_f1={va['macro_f1']:.4f}{mon}")
        print(line)
        if log is not None: log.append(line)

        if best_st is None or (va["macro_f1"] - best_f1) > args.early_stop_min_delta:
            best_f1 = va["macro_f1"]
            best_st = copy.deepcopy(model.state_dict())
        if early.step(va["macro_f1"]):
            msg = f"[{LOG_TAG}] Early stopping at epoch {ep}"
            print(msg)
            if log: log.append(msg)
            break

    if best_st: model.load_state_dict(best_st)
    return model


# ─────────────────────────────────────────────────────────────
# 5. 헬퍼 / 실행
# ─────────────────────────────────────────────────────────────
def _new_model(args, device):
    return HARFNETver3_WO_OA(
        roberta_name=args.roberta, clip_name=args.clip_model).to(device)


def _run_body(model, tl, vl, device, args, log, rep):
    model = train_one_run(model, tl, vl, device, args, log)
    for line in log: rep.write(line + "\n")
    return model


def run_single(csv, root, device, tok, prep, args, freeze, preamble=""):
    labels = base._ds(csv, root, tok, prep, args.max_length, None).labels
    tr, va, te = base.split_602020(labels, args.seed)
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    rep = io.StringIO(); rep.write(preamble)
    rep.write(f"###### {LOG_TAG} (60/20/20) ######\n\n")
    tr_ds = base._ds(csv, root, tok, prep, args.max_length, tr)
    tl = base._dl(tr_ds, args.batch_size, True, args.num_workers, True,
                  sampler=base.make_weighted_sampler(tr_ds.labels, args.sampler_alpha))
    vl = base._dl(base._ds(csv, root, tok, prep, args.max_length, va),
                  args.batch_size, False, args.num_workers)
    el = base._dl(base._ds(csv, root, tok, prep, args.max_length, te),
                  args.batch_size, False, args.num_workers)
    model = _new_model(args, device)
    if freeze: base.freeze_backbones(model)
    log = []; model = _run_body(model, tl, vl, device, args, log, rep)
    yt, yp = base.collect_predictions(model, el, device)
    blk, m = base.eval_report_block(yt, yp, label=LOG_TAG)
    print(blk, end=""); rep.write(blk)
    ts2 = datetime.now().strftime("%Y%m%d_%H%M%S")
    base.save_checkpoint(
        os.path.join(args.checkpoint_dir, f"harfnet_ver3_wo_oa_single_{ts2}.pt"),
        model, args, m)
    path = os.path.join(base.RESULT_DIR, f"harfnet_ver3_wo_oa_single_{ts}.txt")
    with open(path, "w", encoding="utf-8") as f: f.write(rep.getvalue())
    print(f"리포트: {path}")


def run_kfold(csv, root, device, tok, prep, args, freeze, preamble=""):
    labels = base._ds(csv, root, tok, prep, args.max_length, None).labels
    idx = np.arange(len(labels)); y = np.array(labels)
    tva, te, _, _ = train_test_split(
        idx, y, test_size=args.kfold_test_size,
        stratify=y, random_state=args.seed)
    k  = args.kfold_splits
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_rep = io.StringIO(); all_rep.write(preamble)
    all_rep.write(f"###### {LOG_TAG} (k={k}) ######\n")
    el = base._dl(base._ds(csv, root, tok, prep, args.max_length, te.tolist()),
                  args.batch_size, False, args.num_workers)
    skf = StratifiedKFold(n_splits=k, shuffle=base.KFOLD_SHUFFLE,
                          random_state=args.kfold_random_state)
    fold_m = []
    for fold, (str_, sva_) in enumerate(skf.split(np.zeros(len(tva)), y[tva]), 1):
        print(f"\n{'='*70}\n[Fold {fold}/{k}] {LOG_BRAND}\n{'='*70}")
        fd = base._ds(csv, root, tok, prep, args.max_length, tva[str_].tolist())
        tl = base._dl(fd, args.batch_size, True, args.num_workers, True,
                      sampler=base.make_weighted_sampler(fd.labels, args.sampler_alpha))
        vl = base._dl(base._ds(csv, root, tok, prep, args.max_length, tva[sva_].tolist()),
                      args.batch_size, False, args.num_workers)
        model = _new_model(args, device)
        if freeze: base.freeze_backbones(model)
        log = []; fold_buf = io.StringIO()
        model = _run_body(model, tl, vl, device, args, log, fold_buf)
        yt, yp = base.collect_predictions(model, el, device)
        blk, m = base.eval_report_block(yt, yp, label=LOG_TAG)
        print(blk, end=""); fold_buf.write(blk)
        ts2 = datetime.now().strftime("%Y%m%d_%H%M%S")
        base.save_checkpoint(
            os.path.join(args.checkpoint_dir,
                         f"harfnet_ver3_wo_oa_kfold_{ts2}_fold{fold}.pt"),
            model, args, m)
        summ = (f"{LOG_TAG}: 4-way F1={m['f1_4']:.4f} "
                f"H/A={m['f1_ha']:.4f} R/F={m['f1_rf']:.4f}")
        print(f"\n[Fold {fold}] {summ}")
        fold_buf.write(f"\n[Fold {fold}] {summ}\n")
        all_rep.write(f"\n\n{'#'*80}\n### FOLD {fold}/{k}\n{'#'*80}\n")
        all_rep.write(fold_buf.getvalue())
        fold_m.append(m)
        del model; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    def _m(k_): return float(np.mean([r[k_] for r in fold_m]))
    g = (f"{LOG_TAG}: 4-way F1={_m('f1_4'):.4f} "
         f"H/A={_m('f1_ha'):.4f} R/F={_m('f1_rf'):.4f}")
    all_rep.write(f"\n\n{'#'*80}\n### SUMMARY\n{'#'*80}\n{g}\n")
    all_rep.write("Variant\tHA_F1\tRF_F1\t4C_F1\n")
    all_rep.write(f"WO-OA\t{_m('f1_ha'):.4f}\t{_m('f1_rf'):.4f}\t{_m('f1_4'):.4f}\n")
    path = os.path.join(base.RESULT_DIR, f"harfnet_ver3_wo_oa_kfold_{ts}.txt")
    with open(path, "w", encoding="utf-8") as f: f.write(all_rep.getvalue())
    print(f"\n리포트: {path}\n{g}")


# ─────────────────────────────────────────────────────────────
# 6. main
# ─────────────────────────────────────────────────────────────
def main():
    pa = argparse.ArgumentParser(description="HARFNET-VER3 without OverAlignment")
    pa.add_argument("--csv_path",       default=base.DEFAULT_CSV_PATH)
    pa.add_argument("--data_root",      default=base.DEFAULT_DATA_ROOT)
    pa.add_argument("--checkpoint_dir", default=base.CHECKPOINT_DIR)
    pa.add_argument("--batch_size",     type=int,   default=base.BATCH)
    pa.add_argument("--epochs",         type=int,   default=base.EPOCHS)
    pa.add_argument("--lr",             type=float, default=base.LR)
    pa.add_argument("--num_workers",    type=int,   default=base.NUM_WORKERS)
    pa.add_argument("--seed",           type=int,   default=base.SEED)
    pa.add_argument("--early_stop_patience",  type=int,   default=base.EARLY_STOP_PATIENCE)
    pa.add_argument("--early_stop_min_delta", type=float, default=base.EARLY_STOP_MIN_DELTA)
    pa.add_argument("--no_kfold",             action="store_true")
    pa.add_argument("--kfold_splits",         type=int,   default=base.KFOLD_SPLITS)
    pa.add_argument("--kfold_test_size",      type=float, default=base.KFOLD_TEST_SIZE)
    pa.add_argument("--kfold_random_state",   type=int,   default=base.KFOLD_RANDOM_STATE)
    pa.add_argument("--roberta",        default=base.ROBERTA)
    pa.add_argument("--clip_model",     default=base.CLIP_RN101)
    pa.add_argument("--max_length",     type=int,   default=base.MAX_LENGTH)
    pa.add_argument("--sampler_alpha",  type=float, default=0.5)
    pa.add_argument("--lambda_fake_bce", type=float, default=base.LAMBDA_FAKE_BCE)
    pa.add_argument("--lambda_auth_bce", type=float, default=base.LAMBDA_AUTH_BCE)
    pa.add_argument("--lambda_con",      type=float, default=base.LAMBDA_CON)
    pa.add_argument("--no_freeze_encoders", action="store_true")
    pa.add_argument("--no-progress",        action="store_true")
    args = pa.parse_args()

    if args.no_progress: os.environ["VER3_NO_TQDM"] = "1"
    freeze = not args.no_freeze_encoders
    base.set_seed(args.seed)

    preamble = base.loading_preamble(args.csv_path, args.data_root)
    print(f"\n{LOG_BRAND} | KFold={not args.no_kfold} | device={base.DEVICE}")
    print(f"[손실] Focal"
          f" + {args.lambda_fake_bce}·BCE(Fake)"
          f" + {args.lambda_auth_bce}·BCE(Auth)"
          f" + {args.lambda_con}·SupCon")
    print("[Ablation] OverAlignModule / OA loss / AF conditional loss 제거\n")

    tok = base.RobertaTokenizerFast.from_pretrained(args.roberta)
    _, prep = base.openai_clip.load(args.clip_model, device="cpu")

    fn = run_single if args.no_kfold else run_kfold
    fn(args.csv_path, args.data_root, base.DEVICE, tok, prep, args, freeze, preamble)


if __name__ == "__main__":
    main()