import transformers
import logging
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForQuestionAnswering, TrainingArguments, Trainer
from transformers import default_data_collator
import os
import numpy as np
import pandas as pd
os.environ["WANDB_DISABLED"] = "true"

logging.basicConfig(level=logging.DEBUG)


class QADataGen:
    def __init__(self, qa_file, model="distilbert-base-uncased", batch_size=16):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model = model
        self.batch_size = batch_size
        self.n_max_size = 100
        self.n_beam = 30
        self.n_finetune_questions = 100
        self.n_ans = 100
        self.dataset_finetune = load_dataset("squad", split=f"train[0:{self.n_finetune_questions}]")
        self.qa_df = pd.read_csv(qa_file, nrows=self.n_ans)
        self.qa_df.to_csv(f"/qa_{self.n_ans}.csv", index=False)
        self.dataset_final = load_dataset("csv", data_files=f"/qa_{self.n_ans}.csv")
        self.qa_file = qa_file
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.max_length = 384  # The maximum length of a feature (question and context)
        self.doc_stride = 8  # The authorized overlap between two part of the context when splitting it is needed.
        assert isinstance(self.tokenizer, transformers.PreTrainedTokenizerFast)
        model_name = model.split("/")[-1]
        tokenized_datasets = self.dataset_finetune.map(self.prepare_train_features,
                                                       batched=True,
                                                       remove_columns=self.dataset_finetune.column_names)
        args = TrainingArguments(
            f"{model_name}-finetuned-squad",
            evaluation_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=3,
            weight_decay=0.01,
            report_to="none",
            # push_to_hub=True,
        )
        data_collator = default_data_collator
        self.qa_model = AutoModelForQuestionAnswering.from_pretrained(self.model)
        self.trainer = Trainer(
            self.qa_model,
            args,
            train_dataset=tokenized_datasets,
            eval_dataset=tokenized_datasets,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )

    def prepare_train_features(self, examples):
        examples["question"] = [q.lstrip() for q in examples["question"]]

        # Tokenize our examples with truncation and padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        tokenized_examples = self.tokenizer(
            examples["question"],
            examples["context"],
            truncation="only_second",
            max_length=self.max_length,
            stride=self.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        # The offset mappings will give us a map from token to character position in the original context. This will
        # help us compute the start_positions and end_positions.
        offset_mapping = tokenized_examples.pop("offset_mapping")

        # Let's label those examples!
        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []

        for i, offsets in enumerate(offset_mapping):
            # We will label impossible answers with the index of the CLS token.
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(self.tokenizer.cls_token_id)

            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            answers = examples["answers"][sample_index]
            # If no answers are given, set the cls_index as answer.
            if len(answers["answer_start"]) == 0:
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Start/end character index of the answer in the text.
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])

                # Start token index of the current span in the text.
                token_start_index = 0
                while sequence_ids[token_start_index] != 1:
                    token_start_index += 1

                # End token index of the current span in the text.
                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != 1:
                    token_end_index -= 1

                # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
                if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                else:
                    # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                    # Note: we could go after the last offset if the answer is the last word (edge case).
                    while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                        token_start_index += 1
                    tokenized_examples["start_positions"].append(token_start_index - 1)
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples["end_positions"].append(token_end_index + 1)

        return tokenized_examples

    def prepare_validation_features(self, examples):
        # Some of the questions have lots of whitespace on the left, which is not useful and will make the
        # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
        # left whitespace
        examples["question"] = [str(q).lstrip() for q in examples["question"]]
        examples["context"] = [str(c) for c in examples["context"]]

        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        tokenized_examples = self.tokenizer(
            examples["question"],
            examples["context"],
            truncation="only_second",
            max_length=self.max_length,
            stride=self.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

        # We keep the example_id that gave us this feature and we will store the offset mappings.
        tokenized_examples["example_id"] = []

        for i in range(len(tokenized_examples["input_ids"])):
            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(examples["id"][sample_index])

            # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
            # position is part of the context or not.
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == 1 else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]

        return tokenized_examples

    def post_process(self, raw_predictions, validation_features2):
        start_logits = raw_predictions.predictions[0]
        end_logits = raw_predictions.predictions[1]
        # Gather the indices the best start/end logits:
        start_indexes = np.argsort(start_logits, axis=-1)[:, -1: -self.n_beam - 1: -1]
        end_indexes = np.argsort(end_logits, axis=-1)[:, -1: -self.n_beam - 1: -1]
        valid_answers_list = []
        offset_mapping = validation_features2["offset_mapping"]
        # The first feature comes from the first example. For the more general case, we will need to be match the example_id to
        # an example index
        example_ids = validation_features2['example_id']
        context = self.dataset_final["train"]["context"]
        input_ids = validation_features2['input_ids']
        for row in range(start_indexes.shape[0]):
            valid_answers = []
            example_id = example_ids[row]
            for col_start in range(start_indexes.shape[1]):
                for col_end in range(end_indexes.shape[1]):
                    start_index = start_indexes[row, col_start]
                    end_index = end_indexes[row, col_end]
                    if (start_index < end_index) and (start_index < start_logits.shape[1]) and (
                            end_index < end_logits.shape[1]) and (end_index - start_index + 1 <= self.n_max_size):
                        if offset_mapping[row] and offset_mapping[row][start_index] and offset_mapping[row][end_index]:
                            start_char = offset_mapping[row][start_index][0]
                            end_char = offset_mapping[row][end_index][1]
                            context_q = context[example_ids[row]]
                            if end_char < len(context_q):
                                #text = self.tokenizer.decode(input_ids[row][start_char:end_char+1])
                                text = context_q[start_char:end_char + 1]
                                text = text.strip()
                                if len(text):
                                    valid_answers.append(
                                        {
                                            "score": start_logits[row, start_index] + end_logits[row, end_index],
                                            "text": text
                                            # We need to find a way to get back the original substring corresponding to the answer in the context
                                        }
                                    )
            valid_answers = sorted(valid_answers, key=lambda x: x["score"], reverse=True)
            if valid_answers:
                valid_answers[0]['qid'] = example_id
                valid_answers_list.append(valid_answers[0])
            else:
                valid_answers_list.append({'score': -1, 'text': '', 'qid': example_id})
        return valid_answers_list

    def finetune(self, save_model=None):
        self.trainer.train()
        if save_model:
            self.trainer.save_model("test-squad-trained")
        self.logger.info("finetuned underlying QA model")

    def answer_questions(self):
        validation_features2 = self.dataset_final["train"].map(
            self.prepare_validation_features,
            batched=True,
            remove_columns=self.dataset_final["train"].column_names
        )
        raw_predictions = self.trainer.predict(validation_features2)
        validation_features2.set_format(type=validation_features2.format["type"],
                                        columns=list(validation_features2.features.keys()))
        valid_answers_list = self.post_process(raw_predictions, validation_features2)
        answers = [v['text'] for v in valid_answers_list]
        qid = [v['qid'] for v in valid_answers_list]
        scores = [v['score'] for v in valid_answers_list]
        df_ans = pd.DataFrame(data={'question': list(range(len(valid_answers_list))), 'answer': answers, 'qid': qid,
                                    'score': scores})
        return df_ans

    def cb_finetune_file(self, save_ans_file):
        self.finetune()
        df = self.answer_questions()
        df2 = df[["qid", "score"]].groupby(["qid"]).max().reset_index(drop=False)
        df2 = pd.merge(df2, df[["qid", "score", "answer"]], on=["qid", "score"], how="left")
        df2 = df2.groupby(["qid"]).first().reset_index(drop=False)
        qa_df = self.qa_df.drop(columns=["qid"])
        qa_df.rename(columns={"id": "qid"}, inplace=True)
        df2 = pd.merge(df2, self.qa_df[["id", "question"]], on=["qid"], how="inner")
        df2.to_csv(save_ans_file, index=False)


if __name__ == "__main__":
    qa_file = "/final_short.csv"
    savefile = "/answers_qa.csv"
    qa = QADataGen(qa_file=qa_file)
    qa.cb_finetune_file(save_ans_file=savefile)