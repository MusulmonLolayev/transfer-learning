# for image experiments
python3 ./exp_images.py --tr-folder "/home/Musolmon/Documents/AI/datasets/image-classification/CIFAR10/train" --te-folder "/home/Musolmon/Documents/AI/datasets/image-classification/CIFAR10/test" --seed 42 --epochs 5 --output "cifar10.txt" --model efficientnet --w-type nikolay --w-threshold 0 # --only-sign False

# for text experiments
python3 ./exp_texts.py --tr-folder "/home/Musolmon/Documents/AI/datasets/text-classification/gossipcop" --seed 42 --epochs 5 --output "gossipcop.txt" --model small_bert --w-type nikolay --w-threshold 0 # --only-sign False

# for image experiments
python3 ./exp_audio.py --tr-folder "/home/Musolmon/Documents/AI/datasets/audio-classification/mini_speech_commands/"  --seed 42 --epochs 5 --output "mini_speech_commands.txt" --model efficientnet --w-type nikolay --w-threshold 0 # --only-sign False