#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import json
import re
import pandas as pd
from typing import (
    Dict,
    Any
)


class Logger:
    """Check data integrity and compliance"""
    def __init__(self, ds_path: str) -> None:
        # Check if path exists
        if not os.path.exists(ds_path):
            raise FileNotFoundError("Path does not exist.")
        
        # Check if path is a file
        if not os.path.isfile(ds_path):
            raise ValueError("The path is not a file.")
        
        # Identify type of file
        _, ext = os.path.splitext(ds_path)
        self.id = {"format": ext, "path": ds_path}

    def check_illegal_chars(self, pattern: str = None) -> Dict[str, Any]:
        # Check type of file
        if self.id["format"] == '.csv':
            df = pd.read_csv(self.id["path"], encoding="unicode_escape")
        elif self.id["format"] == '.xlsx':
            df = pd.read_excel(self.id["path"])
        else:
            raise ValueError("File must be a 'CSV' or 'Excel'(xlsx) file.")
        
        # Check integrity of column
        required_columns = {'ï»¿instruction', 'user_query', 'context', 'output'}
        missing_columns = required_columns - set(df.columns)

        if missing_columns:
            raise ValueError(f"The following required columns are missing: {missing_columns}")
        
        # Check UTF-8 compliance
        if pattern is None:
            pattern = re.compile(r'[^\x09\x0A\x0D\x20-\uD7FF\uE000-\uFFFD\u10000-\u10FFFF]', re.UNICODE)
        
        for index, row in df.iterrows():
            # Check the contents of context
            context = row.get('context', '')
            illegal_chars = [(m.start(), m.group()) for m in pattern.finditer(context)]
            if illegal_chars:
                highlighted_context = self.__highlight_illegal_chars(context, illegal_chars)
                print("="*100, "\n", f"Illegal characters found in context of line {index}:\n{highlighted_context}")

        return df

    def __highlight_illegal_chars(self, context: str, illegal_chars: list) -> str:
        """Highlight illegal characters and return the processed string"""
        list_context = list(context)
        for pos, char in illegal_chars:
            list_context[pos] = f"\033[91m[***{char}***]\033[0m"  # red highlight

        return ''.join(list_context)
        

class Processor:
    """Column processing of datasets"""
    def __init__(self, dataframe: Any, tokenizer: Any):
        self.df = dataframe
        self.tk = tokenizer

    def calculator(self, *metrics: str) -> None:
        for mc in metrics:
            if mc == "token count":
                self.df[["context length", "token type"]] = self.df.apply(self.__token_count, axis=1, tokenizer=self.tk)
            elif mc == "one hot count":
                self.df[["total count", "one hot count"]] = self.df.apply(self.__non_null_count, axis=1)
            elif mc == "statistic count":
                df_counts = self.df['output'].apply(lambda x: self.__calculate_value_count(x)).apply(pd.Series)
                self.df = pd.concat([self.df, df_counts], axis=1)

    def __token_count(self, row, tokenizer):
        headers = row.index                                                     # Obtain the header of table
        combined_text = " ".join([str(row[header]) for header in headers])      # Concatenate the comma-separated tex
        tokens = tokenizer.tokenize(combined_text)                              # Tokenization
        context_length = len(tokens)
        token_type = len(set(tokens))

        return pd.Series([context_length, token_type])

    def __non_null_count(self, row):
        total_count = 0
        one_hot_count = {
            "T_melt": 0,
            "T_mold": 0,
            "v_inj": 0,
            "p_inj": 0,
            "p_hold": 0,
            "t_hold": 0
        }
        
        if "output" not in row.index:
            raise ValueError("'output' column does not exist in header")
        
        str_json = row.get('output', '')
        try:
            data_dict = json.loads(str_json)
        except json.JSONDecodeError as j:
            return {f"JSONDecodeError: {j}"}  # Return a dictionary indicating the json format error
        except Exception as e:
            return {f"{type(e).__name__}: {e}"}  # Return the dictionary that points out other erroneous contents
        
        for k, v in data_dict.items():
            if v.get('unit') is not None:
                match k:
                    case "melt temperature":
                        one_hot_count["T_melt"] += 1
                    case "mold temperature":
                        one_hot_count["T_mold"] += 1
                    case "injection speed":
                        one_hot_count["v_inj"] += 1
                    case "injection pressure":
                        one_hot_count["p_inj"] += 1
                    case "holding pressure":
                        one_hot_count["p_hold"] += 1
                    case "holding time":
                        one_hot_count["t_hold"] += 1

        total_count = sum(1 for v in data_dict.values() if v.get("unit") is not None)

        return pd.Series([total_count, one_hot_count])    

    def __calculate_value_count(self, str_json):
        output_data = json.loads(str_json)
        value_counts = {
            'T_melt': len(output_data.get('melt temperature', {}).get('value', [])) if isinstance(output_data.get('melt temperature', {}).get('value'), list) else 0,
            'T_mold': len(output_data.get('mold temperature', {}).get('value', [])) if isinstance(output_data.get('mold temperature', {}).get('value'), list) else 0,
            'v_inj': len(output_data.get('injection speed', {}).get('value', [])) if isinstance(output_data.get('injection speed', {}).get('value'), list) else 0,
            'p_inj': len(output_data.get('injection pressure', {}).get('value', [])) if isinstance(output_data.get('injection pressure', {}).get('value'), list) else 0,
            'p_hold': len(output_data.get('holding pressure', {}).get('value', [])) if isinstance(output_data.get('holding pressure', {}).get('value'), list) else 0,
            't_hold': len(output_data.get('holding time', {}).get('value', [])) if isinstance(output_data.get('holding time', {}).get('value'), list) else 0
        }
        return value_counts
    