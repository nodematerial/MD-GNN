#!/bin/bash

if [ "$#" -ne 1 ]; then
  echo "- 引数の数が正しくありません。"
  exit 1
fi

# YAMLファイルのパス
yamlpath="configs/$1"

if [ ! -e "$yamlpath" ]; then
  echo "- 指定されたyamlファイルのパス [$yamlpath] にはファイルが存在しません"
  exit 1
fi

# yqを使用してYAMLファイルを解析し、設定を読み込む
dirname=$(yq eval '.dirname' "$yamlpath")
dumpfile_path="dumpfiles/$dirname"
dataset_path="dataset/$dirname"

# lammpsの実行コマンド
COMMAND1="rm -r $dumpfile_path"
COMMAND2="rm -r $dataset_path"

echo '[started converting logs of physical properties into CSV]'
echo -e "- COMMAND1: $COMMAND1"
echo -e "- COMMAND2: $COMMAND2 \n"
eval $COMMAND1
eval $COMMAND2
echo -e "succeeded"