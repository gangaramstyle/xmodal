"""Sampler-only audit (NO training): does the real BraTS data support the v3 structured sampling?

overlap_rate=0 only proves the exclusion code works. This measures, per track x prism x source-size,
whether the 128 source slots and 12x4 targets actually land on valid, informative anatomy vs zero-
padded / skull-stripped-background / out-of-FOV / near-duplicate positions. Emits a stratified table so
the validity thresholds and register budget are set by data, not guessed.

    python scripts/audit_sampler.py --data-root ~/xmodal/data/brats26 --patients-per-track 30 --bags 1500
"""
from __future__ import annotations
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, torch  # noqa: E402
from xmodal import data as D, sampling as S  # noqa: E402

FG_EPS = 1e-3            # skull-stripped background is exactly 0; padding_mode=zeros is also 0
FG_MIN = 0.25           # a patch is "valid" if >=25% of its voxels are foreground


def frac_fg(patches):
    """[.,V,V,1] -> foreground fraction per patch (mean of voxels > eps)."""
    return (patches.abs() > FG_EPS).float().mean(dim=(-3, -2, -1))


def audit_config(bundles, *, prism, ssize, bags, batch, device, rng):
    n_valid_src, ndup, src_fg, tgt_fg, tgt_allvalid, src_zero, overlap = [], [], [], [], [], [], []
    done = 0
    while done < bags:
        bs = min(batch, bags - done)
        b = S.sample_mixed_paired_batch(
            bundles, batch_size=bs, token_count=128, held_count=48, n_series=8, step=0, total=1,
            patch_sizes=(ssize,), voxels=16, prism_choices=(prism,), orient="native", rng=rng, device=device,
            structured=True, n_pos=12, target_size=8.0, scan_context=False)
        sfg = frac_fg(b["patches_a_reference"])              # [bs,128]
        hfg = frac_fg(b["held_semantic"])                    # [bs,48]
        n_valid_src.append((sfg >= FG_MIN).float().sum(1).cpu().numpy())     # N_real per bag
        src_fg.append(sfg.mean(1).cpu().numpy()); tgt_fg.append(hfg.mean(1).cpu().numpy())
        src_zero.append((sfg < FG_EPS).float().mean(1).cpu().numpy())        # all-zero (padding/bg) source frac
        tgt_allvalid.append(((hfg >= FG_MIN).all(1)).float().cpu().numpy())  # all 48 held valid
        overlap.append(b["overlap_rate"])
        ca = b["coords_a"]                                                   # [bs,128,3] centers rel anchor
        dd = torch.cdist(ca, ca)                                            # in-plane approx; near-dup if < ssize/2
        eye = torch.eye(128, dtype=torch.bool, device=device)[None]
        ndup.append(((dd < ssize / 2) & ~eye).float().sum(-1).mean(-1).cpu().numpy())
        done += bs
    nreal = np.concatenate(n_valid_src)
    return dict(prism=prism, ssize=ssize,
                nreal_med=float(np.median(nreal)), nreal_p5=float(np.percentile(nreal, 5)),
                nreal_lt64=float((nreal < 64).mean()), reg_frac_med=float(np.median((128 - nreal) / 128)),
                src_fg=float(np.mean(np.concatenate(src_fg))), tgt_fg=float(np.mean(np.concatenate(tgt_fg))),
                src_zero=float(np.mean(np.concatenate(src_zero))),
                tgt_allvalid=float(np.mean(np.concatenate(tgt_allvalid))),
                ndup=float(np.mean(np.concatenate(ndup))), overlap=float(np.mean(overlap)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="~/xmodal/data/brats26")
    ap.add_argument("--tracks", nargs="+", default=["mets_train", "ped_train", "goat_gt", "goat_nogt"])
    ap.add_argument("--patients-per-track", type=int, default=30)
    ap.add_argument("--bags", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    rng = np.random.default_rng(0); dev = a.device
    root = os.path.expanduser(a.data_root)
    hdr = (f"{'track':<10} {'prism':>5} {'size':>4} | {'Nreal_med':>9} {'Nreal_p5':>8} {'%<64':>5} "
           f"{'reg%':>5} | {'src_fg':>6} {'tgt_fg':>6} {'src0%':>6} {'tgt_ok':>6} {'ndup':>6} {'ovlp':>5}")
    print(hdr); print("-" * len(hdr))
    for tr in a.tracks:
        d = os.path.join(root, tr)
        if not os.path.isdir(d):
            print(f"{tr:<10} MISSING"); continue
        pid2dir = D.find_brats_patients(d)
        # require complete 4-modality patients (structured needs them)
        comp = {p: dd for p, dd in pid2dir.items()
                if all(__import__("glob").glob(os.path.join(dd, "**", f"*-{suf}.nii.gz"), recursive=True)
                       for suf in D.LOCAL_SUFFIX.values())}
        pids = sorted(comp)[:a.patients_per_track]
        bundles = []
        for p in pids:
            try:
                bundles.append(D.load_local_bundle(p, comp[p], device=dev)[0])
            except Exception:
                continue
        if len(bundles) < 2:
            print(f"{tr:<10} only {len(bundles)} complete bundles"); continue
        for prism in (32.0, 64.0, 128.0):
            for ssize in (4.0, 8.0, 16.0):
                r = audit_config(bundles, prism=prism, ssize=ssize, bags=a.bags, batch=a.batch, device=dev, rng=rng)
                print(f"{tr:<10} {int(prism):>5} {int(ssize):>4} | {r['nreal_med']:>9.0f} {r['nreal_p5']:>8.0f} "
                      f"{r['nreal_lt64']*100:>4.0f}% {r['reg_frac_med']*100:>4.0f}% | {r['src_fg']:>6.2f} {r['tgt_fg']:>6.2f} "
                      f"{r['src_zero']*100:>5.0f}% {r['tgt_allvalid']*100:>5.0f}% {r['ndup']:>6.1f} {r['overlap']:>5.2f}", flush=True)
        del bundles; torch.cuda.empty_cache() if dev == "cuda" else None
    print("\nlegend: Nreal=source patches with >=25% fg; reg%=missing-slot frac; src0%=all-zero source frac; "
          "tgt_ok=bags with all 48 targets valid; ndup=mean #source within size/2; ovlp=target-source overlap")


if __name__ == "__main__":
    main()
