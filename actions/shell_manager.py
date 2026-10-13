import paramiko

from actions.remote_shell import RemoteShell
from config.config import Configs


class ShellManager:
    _instance = None
    _ssh_client = None
    _shell = None
    # Timeout (in seconds) applied to each command execution. Can be overridden
    # before the first connection (e.g. from the CLI) via `ShellManager.command_timeout = ...`.
    command_timeout = 120

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def get_shell(self) -> RemoteShell:
        if self._shell is None:
            self._connect()
        return self._shell

    def _connect(self):
        if self._ssh_client is None:
            self._ssh_client = paramiko.SSHClient()
            self._ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            self._ssh_client.connect(
                hostname=Configs.basic_config.kali['hostname'],
                username=Configs.basic_config.kali['username'],
                password=Configs.basic_config.kali['password'],
                port=Configs.basic_config.kali['port']
            )
        if self._shell is None:
            self._shell = RemoteShell(self._ssh_client.invoke_shell(), command_timeout=self.command_timeout)

    def close(self):
        if self._shell:
            try:
                self._shell.shell.close()
            except:
                pass
            self._shell = None

        if self._ssh_client:
            try:
                self._ssh_client.close()
            except:
                pass
            self._ssh_client = None