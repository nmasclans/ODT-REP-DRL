target_include_directories(odt.x PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

target_sources(odt.x
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/dqn.cc                      ${CMAKE_CURRENT_SOURCE_DIR}/dqn.h
    ${CMAKE_CURRENT_SOURCE_DIR}/model.cc                    ${CMAKE_CURRENT_SOURCE_DIR}/model.h
    ${CMAKE_CURRENT_SOURCE_DIR}/replayMemory.cc             ${CMAKE_CURRENT_SOURCE_DIR}/replayMemory.h
)
