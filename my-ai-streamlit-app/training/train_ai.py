from ai import RealAI  # or whatever your AI file is called

def main():
    ai = RealAI()
    print("Starting training...")
    ai.retrain_on_new_data()  # trains on current data and saves checkpoint
    print("Training complete!")

if __name__ == "__main__":
    main()
