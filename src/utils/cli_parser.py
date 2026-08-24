import argparse
import sys

def get_args():
    parser = argparse.ArgumentParser()
    
    # common arguments for all tasks
    parser.add_argument(
        "--task", 
        type=str, 
        choices=["start_session", "get_status", "query_past", "get_latest_gm"],
        default="start_session",
        help="Task to perform: start a session, get session status, query past session, or get the latest global model"
    )

    parser.add_argument(
        "--server_endpoint",
        type=str,
        default="localhost:12345",
        help="Address of the Federated Learning Server",
    )

    # dynamically read the `--task` argument before fully parsing
    # to find out what else needs to be added to the Parser.
    task = None
    if "--task" in sys.argv:
        try:
            task = sys.argv[sys.argv.index("--task") + 1]
        except IndexError:
            pass
    else:
        for arg in sys.argv:
            if arg.startswith("--task="):
                task = arg.split("=")[1]
                break

    # if --task was not explicitly provided, default to 'start_session'
    if task is None:
        task = 'start_session'

    ## -- ARGUMENTS TO START A SESSION --
    if task == 'start_session':
        parser.add_argument(
            "--config",
            type=str, 
            help="Path to the Federated Learning configuration file",
            required=True
        )
        parser.add_argument(
            "--file",
            action="store_true",
            default=False,
            help="If file flag is provided server will try to restore from a checkpoint file.",
        )
        parser.add_argument(
            "--restore",
            action="store_true",
            default=False,
            help="Restore session by providing the session id.",
        )
        parser.add_argument(
            "--revive",
            action="store_true",
            default=False,
            help="Revive session by providing the session id.",
        )
        
        # additional conditional checks for start_session
        if "--restore" in sys.argv or "--revive" in sys.argv or "--file" in sys.argv:
            parser.add_argument(
                "--session_id",
                type=str,
                required=True,
                help="The id of the session to be restored/revived",
            )

    ## -- ARGUMENTS TO GET STATUS OF A SESSION
    elif task == 'get_status':
        parser.add_argument(
            "--session_id", 
            type=str, 
            help="Session ID to get the status for",
            required=True
        )
    
    ## -- ARGUMENTS TO DOWNLOAD LATESTT GLOBAL MODEL
    elif task == 'get_latest_gm':
        parser.add_argument(
            "--session_id", 
            type=str, 
            help="Session ID to get the global model for",
            required=True
        )

        parser.add_argument(
            "--save_path", 
            type=str, 
            help="Path to save the global model",
            default='./latest_gm'
        )

    ## -- ARGUMENTS TO QUERY A PREVIOUS SESSION
    elif task == 'query_past':
        parser.add_argument(
            "--session_id", 
            type=str, 
            help="Session ID to query for",
            required=True
        )


    return parser.parse_args()