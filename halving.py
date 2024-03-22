from datetime import datetime, timedelta

# Input data
current_block_height = 835745  # Enter the current block height
last_halving_block_height = 630000  # Enter the block height at the last halving
blocks_per_halving = 210000
average_block_time_minutes = 10  # Average time to mine a block

# Calculate blocks until the next halving
blocks_until_next_halving = blocks_per_halving - (current_block_height - last_halving_block_height)

# Calculate time until next halving
minutes_until_next_halving = blocks_until_next_halving * average_block_time_minutes
days_until_next_halving = minutes_until_next_halving / 60 / 24

# Calculate the estimated date of the next halving
estimated_date_of_next_halving = datetime.now() + timedelta(days=days_until_next_halving)

print("Estimated date of the next Bitcoin halving:", estimated_date_of_next_halving.strftime("%Y-%m-%d %H:%M:%S"))
