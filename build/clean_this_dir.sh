#!/bin/bash

shopt -s extglob

rm -vrf !("README.md"|"user_config"|"user_config_dockerContainer"|"clean_this_dir.sh")
