import re
import time
from typing import Optional
import paramiko


class SSHOutputHandler:
    """Handles SSH output processing with improved encoding detection and buffering."""

    ENCODINGS = ['utf-8', 'latin-1', 'cp1252', 'ascii']
    BUFFER_SIZE = 8192

    @staticmethod
    def decode_output(data: bytes) -> str:
        """Attempts to decode byte data using multiple encodings."""
        for encoding in SSHOutputHandler.ENCODINGS:
            try:
                return data.decode(encoding)
            except UnicodeDecodeError:
                continue
        return data.decode('utf-8', errors='replace')

    @staticmethod
    def receive_data(shell: paramiko.Channel, timeout: float) -> str:
        """Receives data from shell with improved timeout handling."""
        start_time = time.time()
        retries = 0
        out = ""

        while True:
            if shell.recv_ready():
                data = shell.recv(SSHOutputHandler.BUFFER_SIZE)
                decoded_data = SSHOutputHandler.decode_output(data)
                out += decoded_data

            # Split the output into lines and clean up empty lines
            lines = out.split('\n')
            lines = [x.strip() for x in lines if x.strip() != '']

            if len(lines) > 0:
                last_line = lines[-1].strip()

                # Handle sudo mode detection
                if 'sudo' in last_line:
                    retries += 1

                # Check for common shell prompt formats
                if ('@' in last_line and (last_line[-1] == '$' or last_line[-1] == '#')) or \
                        ('bash' in last_line and (last_line[-1] == '$' or last_line[-1] == '#')):
                    break
                elif last_line[-1] in ['?', '$', '#'] or \
                        '--more--' in last_line.lower():
                    retries += 1
                elif last_line[-1] == ':' and '::' not in last_line and '-->' not in last_line:
                    retries += 1
                elif last_line[-1] == '>' and '<' not in last_line and '-->' not in last_line:
                    retries += 1
                elif any(pattern in last_line.lower() for pattern in ['[y/n]', '[Y/n/q]', 'yes/no/[fingerprint]', '(yes/no)']):
                    retries += 1
                elif 'What do you want to do about modified configuration file sshd_config?' in out:
                    break  # Special case handling for configuration prompts

                # Stop if retries exceed threshold
                if retries >= 3:
                    break

            # Check for timeout
            if time.time() - start_time > timeout:
                shell.send('\x03')
                break

            # Continuously receive more data from the shell
            time.sleep(0.1)  # Small delay to avoid tight loop

        return out  # Return whatever has been received so far


class RemoteShell:
    """Enhanced remote shell handler with improved prompt detection and command execution."""

    FORBIDDEN_COMMANDS = {'apt', 'apt-get'}

    def __init__(self, shell: paramiko.Channel, timeout: float = 120.0, command_timeout: float = 120.0):
        self.shell = shell
        self.command_timeout = command_timeout
        self._setup_shell(timeout)

    def _setup_shell(self, timeout: float) -> None:
        """Initializes shell settings."""
        try:
            self.shell.settimeout(timeout)
            self.shell.set_combine_stderr(True)

            # Create .hushlogin to suppress the welcome message
            self.execute_cmd("touch ~/.hushlogin")

            # Update MOTD settings to disable welcome banners
            motd_commands = [
                "sudo touch /etc/legal",  # Empty legal notice
                "sudo chmod 644 /etc/legal",
                "sudo rm -f /etc/motd",  # Remove message of the day
                "sudo rm -f /etc/update-motd.d/*"  # Remove dynamic MOTD scripts
            ]

            for cmd in motd_commands:
                self.execute_cmd(cmd)

        except Exception as e:
            print(f"Shell setup warning: {e}")

    def _check_forbidden_commands(self, cmd: str) -> Optional[str]:
        """Validates command against forbidden commands list."""
        cmd_parts = cmd.split()
        if any(cmd in self.FORBIDDEN_COMMANDS for cmd in cmd_parts):
            return "Command not allowed: network tunneling tools are restricted"
        return None

    def execute_cmd(self, cmd: str) -> str:
        """Executes command with improved output handling and error recovery."""
        if error_msg := self._check_forbidden_commands(cmd):
            return error_msg

        self.shell.send(cmd + '\n')

        output = self._handle_normal_execution()

        final_output = ''.join(output)

        if "dirb" in cmd and "gobuster" not in cmd:
            return clean_dirb_output(final_output)

        elif "msfconsole" in cmd:
            return clean_msfconsole_output(final_output)

        return final_output

    def _handle_normal_execution(self) -> list:
        """Handles normal command execution flow."""
        output = []

        time.sleep(.5)
        data = SSHOutputHandler.receive_data(self.shell, timeout=self.command_timeout)
        if data != '':
            output.append(data)
        last_line = data.strip().split('\n')[-1]

        if any(pattern in last_line.lower() for pattern in
               ['[y/n]', '[Y/n/q]', 'yes/no/[fingerprint]', '(yes/no)']):
            self.shell.send("yes\n")
            time.sleep(.5)
            data = SSHOutputHandler.receive_data(self.shell, timeout=self.command_timeout)
            if data != '':
                output.append(data)

        return output


def clean_dirb_output(output):
    """Clean the output from the 'dirb' command."""
    # Remove ANSI escape sequences (for colored output)
    output = re.sub(r'\x1b\[[0-9;]*[mGKH]', '', output)

    # Extract initial summary lines (URL_BASE, WORDLIST_FILES, etc.)
    summary_pattern = r"(URL_BASE:.*\n|WORDLIST_FILES:.*\n|GENERATED WORDS:.*\n|---- Scanning URL:.*\n)"
    summary = "\n".join(re.findall(summary_pattern, output))

    # Extract the discovered URLs (lines with CODE and SIZE)
    url_pattern = r"http[^\s]+ \(CODE:[0-9]+\|SIZE:[0-9]+\)"
    urls = "\n".join(re.findall(url_pattern, output))

    # Extract the final downloaded and found information
    stats_pattern = r"DOWNLOADED: \d+ - FOUND: \d+"
    stats = "\n".join(re.findall(stats_pattern, output))

    # Combine summary, URLs, and stats
    cleaned_output = f"{summary}\n{urls}\n{stats}"

    return cleaned_output


def clean_msfconsole_output(output):
    """Clean the output from the 'msfconsole' command."""

    output = re.sub(r'\x1b\[[0-9;]*[mGKH]', '', output)

    cleaned = [line for line in output.splitlines() if
               not any(x in line.lower() for x in
                       ["loading", "warning:", "starting", "====="])]

    # Keep relevant parts such as version info, exploits, payloads, and prompt
    relevant_output = []
    for line in cleaned:
        if line.strip() and any(
                keyword in line.lower() for keyword in ['metasploit', 'exploits', 'payloads', ' >', '-', '+']):
            relevant_output.append(line.strip())

    # Join the relevant output
    return "\n".join(cleaned)
