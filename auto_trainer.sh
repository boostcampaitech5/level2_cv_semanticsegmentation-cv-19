while true
do
    pgrep -f auto_trainer.py || python auto_trainer.py
    sleep 10
done