target_include_directories(odt.x PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

target_sources(odt.x
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/dqnImpl.cc                  ${CMAKE_CURRENT_SOURCE_DIR}/dqnImpl.h
    ${CMAKE_CURRENT_SOURCE_DIR}/model.cc                    ${CMAKE_CURRENT_SOURCE_DIR}/model.h
    ${CMAKE_CURRENT_SOURCE_DIR}/replayMemory.cc             ${CMAKE_CURRENT_SOURCE_DIR}/replayMemory.h
    ${CMAKE_CURRENT_SOURCE_DIR}/reinforcementLearning.cc    ${CMAKE_CURRENT_SOURCE_DIR}/reinforcementLearning.h
)
