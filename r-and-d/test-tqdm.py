from random import random
from tqdm import tqdm
import time

for i in tqdm(range(1000)):
    # random sleep
    time.sleep(random() * 0.1)
