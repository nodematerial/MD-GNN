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
logpath="lmpfile/$dirname/log.lammps"
thermopath="dataset/$dirname/thermo.csv"

if [ ! -e "$logpath" ]; then
  echo "- 指定されたlammps実行ファイルのパス [$lmppath] にはファイルが存在しません"
  exit 1
fi

# lammpsの実行コマンド
COMMAND="poetry run python3 script/log_process.py $logpath $thermopath "

echo '[started converting logs of physical properties into CSV]'
echo -e "- COMMAND: $COMMAND \n"
eval $COMMAND
echo -e "succeeded"