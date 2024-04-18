# for image experiments
python3 ./exp_images.py --tr-folder "path/to/dataset" --te-folder "path/to/dataset" --seed 42 --epochs 5 --output "cifar10.txt" --model efficientnet --w-type nikolay --w-threshold 0 --optimizer Adam # --only-sign

# for text experiments
python3 ./exp_texts.py --tr-folder "path/to/dataset" --seed 42 --epochs 5 --output "gossipcop.txt" --model small_bert --w-type nikolay --w-threshold 0 # --only-sign

# for image experiments
python3 ./exp_audio.py --tr-folder "path/to/dataset"  --seed 42 --epochs 5 --output "mini_speech_commands.txt" --model efficientnet --w-type nikolay --w-threshold 0 # --only-sign