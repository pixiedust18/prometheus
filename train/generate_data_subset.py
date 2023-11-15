from datasets import load_dataset, concatenate_datasets, interleave_datasets

dataset = load_dataset("kaist-ai/Feedback-Collection", split="train")

contain_consistent = dataset.filter(lambda example: "consistent" in example["orig_criteria"])
contain_comprehensive = dataset.filter(lambda example: "comprehensive" in example["orig_criteria"])
contain_correct = dataset.filter(lambda example: "correct" in example["orig_criteria"])
contain_useful = dataset.filter(lambda example: "useful" in example["orig_criteria"])

print(len(dataset)) # 99,952
print(len(contain_consistent), len(contain_comprehensive), len(contain_correct), len(contain_useful))
# 2195 1795 5000 800

filtered_list = [contain_consistent, contain_comprehensive, contain_correct, contain_useful]
filtered = concatenate_datasets(filtered_list)
print(len(filtered)) # 9790

seed = 42
probabilities = [0.25, 0.25, 0.25, 0.25]
filtered_unsample = interleave_datasets(filtered_list, probabilities=probabilities, seed=seed) 
print(len(filtered_unsample)) # 3110

filtered.to_json("datasubset/filtered10k.json")
filtered_unsample.to_json("datasubset/filtered3k.json")

# transform a jsonl file to a json file with square bracket [] and comma ,
def jsonl_to_json(filename):
    with open(f"{filename}.json", "r") as f:
        lines = f.readlines()
        print(len(lines))
        lines[0] = "[" + lines[0]
        lines[-1] = lines[-1] + "]"
        lines = [line + "," for line in lines]
        lines[-1] = lines[-1][:-1]
        print(len(lines))
        print(lines[-1])
        with open(f"json_{filename}.json", "w") as f:
            f.writelines(lines)

jsonl_to_json("filtered10k")
jsonl_to_json("filtered3k")