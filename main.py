import argparse
import torch
from torch import nn
from torch.utils.data import DataLoader
from dataset.data_loader.dataset_gec import load_data, save_data
from model.muti_model import MultiModel

def train(args, train_dataloader, val_dataloader):
    model = MultiModel(args).to(device=args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_func = nn.CrossEntropyLoss()
    num_epochs = args.epoch
    best_accuracy = 0.0
    for epoch in range(1, num_epochs + 1):
        train_l_sum, train_acc_sum, n = 0., 0., 0
        for i, batch in enumerate(train_dataloader):
            _, batch_text, batch_img, y = batch
            batch_text = batch_text.to(device=args.device)
            batch_img = batch_img.to(device=args.device)
            y = y.to(device=args.device)
            y_hat_text, y_hat_img, y_hat = model(batch_text=batch_text, batch_img=batch_img)

            if y_hat is not None:
                loss = loss_func(y_hat, y.long()).sum() + 0.2 * (loss_func(y_hat_text, y.long()).sum() +  loss_func(y_hat_img, y.long()).sum())
            elif y_hat_text is not None and y_hat_img is None:
                loss = loss_func(y_hat_text, y.long()).sum()
            elif y_hat_text is None and y_hat_img is not None:
                loss = loss_func(y_hat_img, y.long()).sum()
            else:
                continue

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_l_sum += loss.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).float().sum().item()
            n += y.shape[0]
        print('epoch %d, loss %.4f, train acc %.3f' % (epoch, train_l_sum / n, train_acc_sum / n))

        accuracy = evaluate(args, model, val_dataloader, epoch)
        if best_accuracy < accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), args.checkpoints_dir + 'best' + '-checkpoint.pth')
    torch.save(model.state_dict(), args.checkpoints_dir + '/final_checkpoint.pth')


def evaluate(args, model, val_dataloader, epoch=None):
    val_acc_sum_text, val_acc_sum_img, val_acc_sum, val_acc_sum_ensemble, n = 0., 0., 0., 0., 0
    for i, batch in enumerate(val_dataloader):
        _, batch_text, batch_img, y = batch
        batch_text = batch_text.to(device=args.device)
        batch_img = batch_img.to(device=args.device)
        y = y.to(device=args.device)
        y_hat_text, y_hat_img, y_hat = model(batch_text=batch_text, batch_img=batch_img)

        val_acc_sum_text += (y_hat_text.argmax(dim=1) == y).float().sum().item()
        val_acc_sum_img += (y_hat_img.argmax(dim=1) == y).float().sum().item()
        val_acc_sum += (y_hat.argmax(dim=1) == y).float().sum().item()
        n += y.shape[0]

    if epoch:
        print('epoch %d, val acc %.4f(text), val acc %.4f(image), val acc %.4f(fusion)'
              % (epoch, val_acc_sum_text / n, val_acc_sum_img / n, val_acc_sum / n))
    else:
        print('val acc %.4f(text), val acc %.4f(image), val acc %.4f(fusion)'
              % (val_acc_sum_text / n, val_acc_sum_img / n, val_acc_sum / n))
    return val_acc_sum / n


def test(args, test_dataloader, dev_dataloader):
    model = MultiModel(args).to(device=args.device)
    model.load_state_dict(torch.load(args.checkpoints_dir + '/final_checkpoint.pth'))
    evaluate(args, model, dev_dataloader)

    predicts = []
    for _, batch in enumerate(test_dataloader):
        ids, batch_text, batch_img, _ = batch
        batch_text = batch_text.to(device=args.device)
        batch_img = batch_img.to(device=args.device)
        y_hat_text, y_hat_img, y_hat = model(batch_text=batch_text, batch_img=batch_img)
        pre = y_hat.argmax(dim=1)
        for i in range(len(ids)):
            guid = ids[i]
            tag = test_dataloader.dataset.label_dict_str[int(pre[i])]
            content = {
                'guid': guid,
                'tag': tag,
            }
            predicts.append(content)
    save_data(args.pre_file, predicts)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--do_train', action='store_true', help='执行训练')
    parser.add_argument('--do_test', action='store_true', help='执行测试')

    parser.add_argument("-lr", "--lr", type=float, default=1e-5)
    parser.add_argument("-dropout", "--dropout", type=float, default=0.0)
    parser.add_argument("-epoch", "--epoch", type=int, default=10)
    parser.add_argument("-batch_size", "--batch_size", type=int, default=4)
    parser.add_argument("--img_size", "--img_size", type=int, default=384)
    parser.add_argument("--text_size", "--text_size", type=int, default=64)
    parser.add_argument("--fuse_strategy", "--fuse_strategy", type=str, default='transformer', help='融合策略')
    arguments = parser.parse_args()
    arguments.train_file = './dataset/train.json'
    arguments.val_file = './dataset/val.json'
    arguments.test_file = './dataset/test.json'
    arguments.pre_file = './predict.txt'
    arguments.checkpoints_dir = './checkpoints'
    arguments.pretrained = 'xlm-roberta-base'
    arguments.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if arguments.do_train:
        train_set, val_set = load_data(arguments)
        train_dataloader_ = DataLoader(train_set, shuffle=True, batch_size=arguments.batch_size)
        eval_dataloader_ = DataLoader(val_set, shuffle=True, batch_size=arguments.batch_size)
        print('model training...')
        train(arguments, train_dataloader_, eval_dataloader_)
        
    if arguments.do_test:
        test_set, val_set = load_data(arguments)
        test_dataloader_ = DataLoader(test_set, shuffle=False, batch_size=arguments.batch_size)
        eval_dataloader_ = DataLoader(val_set, shuffle=False, batch_size=arguments.batch_size)
        print('model testing...')
        test(arguments, test_dataloader_, eval_dataloader_)