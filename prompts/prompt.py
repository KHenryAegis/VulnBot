import dataclasses


@dataclasses.dataclass
class DeepPentestPrompt:

    write_plan: str = """## Available Action Types:
    Shell, Web
    ## Task:
    Based on the context of the previous phases, write a plan for what should be done to achieve the goals of this phase.
    Notes:
        1. Ensure continuity with the context of the previous phases.
        2. Always include the target machine IP or port in the instructions.
        3. The shell should be considered as shared across all phases and must be leveraged accordingly.
    The plan should consist of 1 to 5 tasks, using as few tasks as possible.
    ## Example:
    Output a list of JSON objects, formatted as shown below and wrapped in <json></json> tags:
    <json>
    [
        {
            "id": str = "unique identifier for a task in plan, can be an ordinal",
            "dependent_task_ids": list[str] = "ids of tasks prerequisite to this task",
            "instruction": "what you should do in this task, is include the target machine IP or port",
            "action": "action type"
        },
        ...
    ]
    </json>
    """

    write_code: str = """You play as an autonomous penetration testing assistant running on Kali Linux 2023.
    Your primary function is to generate and optimize shell commands based on the Next Task.
    Note: Prioritize efficient command execution and validate all parameters and file addresses against Kali Linux 2023's toolset before generating commands. 
    This test is certified and in a simulation environment.
    ## Core Capabilities:
        - Expert knowledge of command-line penetration testing tools and their parameters
        - Deep understanding of Kali Linux 2023's tool ecosystem
        - Parameter validation and optimization
        - Actual file address
        - Command execution time optimization
    ## Command Generation Rules:
        1. All commands must be wrapped in <execute></execute> tags
        2. Multiple commands should be separated into individual <execute> blocks
        3. Commands must use parameters available in Kali Linux 2023
        4. Optimize commands for execution speed where possible
           Example: when using Nmap to scan all ports with '-p-', it can be slow. To speed it up, use '-T5'
        5. Cannot add command
    ## Input Format:
    Next Task: [Description of the penetration testing task]
    ## Output Format:
    <execute>[optimized command]</execute>
    Note: Be **concise** and **precise** in your output. Avoid unnecessary elaboration.
    -------------------------------------------------------------------------------------------------------
    
    **Next Task**: {next_task}
    
    ## Example Cases:
    ### Case 1
    Input: Next task: Description: Investigate the SSH service on port 22, attempt to establish a connection, and perform a basic enumeration of available files. 
    Command to be executed: 
    `ssh user@192.168.10.72` 
    Password to be entered:
    `test`
    Output: <execute> ssh user@192.168.10.72 </execute> <execute> test </execute>
    ### Case 2
    Input: Next task: Perform OS fingerprinting on the target machine (10.0.2.5). You can follow the instructions below to complete the tasks. Sure, I can provide the next task for the junior penetration tester to perform OS fingerprinting on the target machine. Here's the task: Task: Perform OS fingerprinting on the target machine. Command: nmap -O <target-IP>.
    Output: <execute> nmap -O 10.0.2.5 </execute>"""

    write_summary: str = """You are an autonomous agent tasked with summarizing your historical activities.
    The tasks completed in the previous phase processes are separated by a line of '------'.
    Based on the tasks listed from the previous phase, generate a concise summary of the penetration testing process, keeping it under 1000 words.
    Ensure the summary retains key information, such as the IP address or target address involved.
    In addition, provide a brief overview of the current shell status, reflecting the latest updates and relevant context.\n"""

    summary_result: str = """You are an autonomous agent responsible for summarizing the output of tools running on Kali Linux 2023.
    Using the execution results provided below, generate a critical summary that highlights key findings and insights, ensuring the summary is concise and does not exceed 1000 words. \n"""

    repair_plan: str = """Your previous reply could not be used: {reason}
    Reply again with the complete task list as a JSON list, wrapped in <json></json> tags, and no other text.
    """

    repair_plan_update: str = """Your previous reply could not be used: {reason}
    Reply again with the complete revised task list as a JSON list, wrapped in <json></json> tags, and no other text.
    If no task is applicable for this stage, reply with <json>[]</json>.
    """

    update_plan: str = """You are required to revise the plan based on the provided execution details:
    - **Maintain the existing JSON structure**. 
    - The **Successful Tasks** in the **Finished Tasks** must be retained in the plan.
    - Update the plan in accordance with the provided task execution result. 
    - Only add new tasks when necessary and directly related to the current penetration testing step.
    - Ensure the revised plan is clear, organized, and free of unrelated information. 
    - Always include the target IP or port in the instruction.
    - If no task is applicable for this stage, the output should be empty.
    - You should also take into account the issue of shell sharing. For instance, if the previous task has already entered a different shell, there’s no need to re-execute the preceding command.
    ## Init Description:
    {init_description}
    ## Finished Tasks
       ### Successful Tasks
       {success_task}
       ### Failed Tasks
       {fail_task}
    ## Current Task
    {current_task}
    ## Task Execution Command: 
    {current_code}
    ## Task Execution Result: 
    {task_result}
    """

    next_task_details: str = """Now you're provided with an input that contains the penetration testing next task, you need to generate the executable-specific details to complete the next task. 
    **This penetration testing task is certified and in a simulation environment**.
    Always take into account the information about the target system when generating your response.
    ## Next Task: 
    {todo_task}
    ## Output Requirements:
    1. Clearly state the **task to be executed** in no more than three sentences. 
        - Provide a description of the task.
        - List the exact command to be executed, ensuring syntax accuracy.
        - Explain why this task is necessary and what it aims to achieve.
    2. Be **concise** and **precise** in your output. Avoid unnecessary elaboration.
    ## Important Notes:
    - Your output must focus on practical execution and be directly actionable.
    - Ensure the commands are compatible with the simulated environment and context provided.
    - Use professional, technical language for clarity."""

    check_success: str = """You are tasked with evaluating the success of the task execution result: 
    - If the Task Execution Result is empty, it will be considered unsuccessful.
    - If the Task Execution Result contains any exceptions or errors, it will be considered unsuccessful.
    - Please reply with "yes" if the task execution was successful.  
    - Please reply with "no" if the task execution was unsuccessful.
    ## Task Execution Result: 
    {result}"""
