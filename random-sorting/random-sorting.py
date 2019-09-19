import random

numbers = random.sample(range(100), 4)

print("Original list: ",  numbers)

max_iter = 1000
current_iter = 0

def random_sort(numbers):

while current_iter < max_iter:

  random.shuffle(numbers)

  is_ordered_ascending = all(numbers[i] <= numbers[i+1] for i in range(len(numbers)-1))
  is_ordered_descending = all(numbers[i] >= numbers[i+1] for i in range(len(numbers)-1))

  print("After {} shuffle(s): {}".format(current_iter + 1, numbers))
  print("Is in ascending order: ", is_ordered_ascending)
  print("Is in descending order: ", is_ordered_descending)

  if is_ordered_ascending or is_ordered_descending:
    break

  current_iter = current_iter + 1

