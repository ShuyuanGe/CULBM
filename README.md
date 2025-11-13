# Introduction

## Implementation Detail

1. Naming Conventions

- namespace
    | 项目 | 命名规则 | 示例 |
    |:---:|:-------:|:---:|
    |风格|全小写，下划线分隔不同单词|namespace fast_lbm {}|
    |分层结构|project::level_1::level_2|namespace project::module_level_1::module_level_2 {}|
    |非外部可见实现|module::detail|namespace module::detail {}|

- class, struct, enum
    | 项目 | 命名规则 | 示例 |
    |:---:|:---:|:----:|
    |风格|首字母大写，每个单词首字母大写|class LbmSolver, struct Vec3|

- function
    | 项目 | 命名规则 | 示例 |
    |:---:|:---:|:----:|
    |风格|非首单词首字母大写，其余字母均小写|initDevice|

- variable
    |项目|命名规则|示例|
    |:---:|:--------:|:---:|
    |常量|全大写，单词之间用下划线连接|MAX_ITER|
    |成员变量|以下划线开头，非首单词首字母大写，其余均小写|_data|
