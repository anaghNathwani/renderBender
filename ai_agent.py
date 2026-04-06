class AIAgent:
    def __init__(self):
        self.instructions = []

    def accept_instruction(self, instruction: str):
        self.instructions.append(instruction)
        print(f"Instruction received: {instruction}")

    def train(self):
        if not self.instructions:
            print("No instructions to train on.")
            return
        print("Training with the following instructions:")
        for instr in self.instructions:
            print(f"- {instr}")

# Example usage:
if __name__ == '__main__':
    agent = AIAgent()
    while True:
        instruction = input("Enter instruction for AI agent (or 'exit' to quit): ")
        if instruction.lower() == 'exit':
            break
        agent.accept_instruction(instruction)
    agent.train()