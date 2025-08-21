project('vector-db', 'cpp')

cpr_dep = dependency('cpr')
asio_dep = dependency('asio')
thread_dep = dependency('threads')
jsoncpp_dep = dependency('nlohmann_json')

executable(
  'backend',
  'src/main.cpp',
  include_directories: ['include'],
  dependencies: [thread_dep, asio_dep, cpr_dep, jsoncpp_dep],
)
