import random
import pandas as pd

# Define a list of appliances
appliances = [
    "fan", "light", "heater", "air conditioner", "television", "oven",
    "washing machine", "refrigerator", "coffee maker", "microwave",
    "speaker", "robot vacuum", "humidifier", "electric kettle", "pump"
]

# Define expanded command phrases with intent labels
# 0: "Turn on appliance"
# 1: "Turn off appliance"
# 2: "Set timer for appliance"
# 3: "Map appliance to port"
command_phrases = {
    # Turn on commands
    0: [
        "turn on the", "switch on the", "activate the", "power up the", 
        "start the", "enable the", "please turn on the", 
        "could you turn on the", "I'd like to activate the", "initiate the"
    ],
    # Turn off commands
    1: [
        "turn off the", "switch off the", "deactivate the", "power down the", 
        "stop the", "disable the", "please turn off the", 
        "could you turn off the", "I'd like to deactivate the", "shut down the"
    ],
    # Set timer commands
    2: [
        "set a timer for the", "schedule the", "start a timer for the", 
        "begin a timer for the", "configure the timer for the", 
        "set up a timer for the", "please set a timer for the", 
        "I'd like to schedule the", "initiate a timer for the", 
        "could you set a timer for the"
    ],
    # Map appliance to port commands
    3: [
        "map the", "assign the", "link the", "connect the", 
        "configure the", "associate the", "please map the", 
        "I'd like to link the", "could you connect the", "pair the"
    ]
}

# Create an empty list to hold the dataset
dataset = []

# Generate samples with noise and variation
for appliance in appliances:
    for intent_label, phrases in command_phrases.items():
        for phrase in phrases:
            for i in range(5):  # Create 5 variations per command for basic balancing
                # Optionally add some noise: extra words, slight variations
                noise_options = [
                    "", "now", "right away", "immediately", "as soon as possible", 
                    "at your earliest convenience", "kindly"
                ]
                noise = random.choice(noise_options)
                # Construct the command
                command = f"{phrase} {appliance} {noise}".strip()
                # Append the command and label to the dataset
                dataset.append({"text": command, "label": intent_label})

# Ensure class balancing by generating additional samples for underrepresented classes
max_samples_per_class = max([len(dataset) // 4, 100])  # Ensure at least 100 per class

balanced_dataset = []
for intent_label in range(4):
    class_samples = [sample for sample in dataset if sample['label'] == intent_label]
    while len(class_samples) < max_samples_per_class:
        # Duplicate and slightly vary existing samples to balance the dataset
        sample = random.choice(class_samples)
        text = sample['text']
        # Add slight variation (e.g., different word order or adding/removing noise)
        if random.random() < 0.3:
            text = text.replace("the", "")  # Randomly remove "the" for variation
        balanced_dataset.append({"text": text.strip(), "label": intent_label})
        class_samples.append(sample)
    balanced_dataset.extend(class_samples[:max_samples_per_class])

# Shuffle the final balanced dataset
final_df = pd.DataFrame(balanced_dataset).sample(frac=1).reset_index(drop=True)

# Save the final dataset to a CSV file
final_df.to_csv("balanced_appliance_commands_dataset.csv", index=False)

print("Balanced synthetic dataset created with", len(final_df), "samples.")
