import argparse, torch, torch.nn as nn
from torch.utils.data import DataLoader, random_split
from models import MultiTaskResNet50
from data import ActionPersonDataset
from utils import save_ckpt, action_accuracy, person_f1
from tqdm import tqdm

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--labels_csv", required=True)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--pos_weight", type=float, default=1.0)
    args = ap.parse_args()

    full = ActionPersonDataset(args.data_root, args.labels_csv, args.img_size, True)
    n_val = max(1, int(0.2*len(full)))
    train_ds, val_ds = random_split(full, [len(full)-n_val, n_val])
    train_ds.dataset.is_train = True
    val_ds.dataset.is_train = False

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MultiTaskResNet50(num_actions=len(full.actions)).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
    ce = nn.CrossEntropyLoss()
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([args.pos_weight], device=device))

    for epoch in range(1, args.epochs+1):
        model.train()
        for x, ya, yp in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):
            x, ya, yp = x.to(device), ya.to(device), yp.to(device)
            optim.zero_grad()
            la, lp = model(x)
            loss = ce(la, ya) + bce(lp, yp)
            loss.backward(); optim.step()

        # validation
        model.eval(); ta=tp=c=0; fsum=n=0
        with torch.no_grad():
            for x, ya, yp in val_loader:
                x, ya, yp = x.to(device), ya.to(device), yp.to(device)
                la, lp = model(x)
                ta += action_accuracy(la, ya)*len(x)
                fsum += person_f1(lp, yp)*len(x)
                c += len(x); n += 1
        print(f"[VAL] action_acc={ta/c:.3f} person_f1={fsum/c:.3f}")
        save_ckpt(model, f"checkpoints/mtl_resnet50_e{epoch}.pt")

if __name__ == "__main__": main()
