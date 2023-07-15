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

# lammpsの実行コマンド
COMMAND="poetry run python3 script/cna.py $yamlpath"

echo '[started converting atom positions into graph]'
echo -e "- COMMAND: $COMMAND \n"
eval $COMMAND
echo -e "succeeded"