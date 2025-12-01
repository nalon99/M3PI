# Internal Technical Support Knowledge Base (FAQ)
This knowledge base provides answers to frequently asked technical questions related to hardware, network access, software, and account management.

## Computer and Hardware Issues
### Startup and Performance  
1. Q: My laptop won't power on. What should I check first? A: First, ensure the power adapter is securely plugged into both the laptop and a working wall outlet. Check the power indicator light on the adapter. If the light is off, try a different outlet. If the light is on but the laptop remains unresponsive, hold the power button down for 30 seconds to perform a hard reset, then attempt to power on again.

2. Q: My computer is running very slowly. How can I troubleshoot performance issues? A: Open the Task Manager (Ctrl+Shift+Esc) and check the 'Processes' tab. Look for applications consuming unusually high percentages of CPU or Memory. Close unnecessary programs. If the issue persists, reboot your machine. If the slowness continues after a reboot, clear your browser cache and cookies, and run the built-in system cleaner utility.

3. Q: The monitor is blank, but the computer seems to be running. What is the fix? A: Check the monitor's power cable and the video cable (HDMI/DisplayPort) connection at both the monitor and the computer. Try pressing the 'Input' button on the monitor to cycle through available video sources (e.g., HDMI 1, DisplayPort). If possible, test the monitor with another computer or the computer with another monitor to isolate the faulty component.

4. Q: How do I request a hardware upgrade (e.g., more RAM)? A: Hardware upgrade requests must be submitted via the 'IT Service Request' portal. Select the category 'Hardware Modification' and provide a detailed business justification for the upgrade, including the specific software applications causing performance bottlenecks. Requests are reviewed bi-weekly.

### Peripherals and Docking
5. Q: My USB device (e.g., mouse, external drive) is not recognized. A: Try plugging the device into a different USB port on your computer or docking station. If using a docking station, try connecting the device directly to the laptop. Check Device Manager for unrecognized devices (look for yellow exclamation marks). Restarting the computer often resolves temporary driver issues.

6. Q: The sound/audio on my headphones is not working. A: Check the volume levels in the system tray and ensure the correct output device is selected. For Bluetooth devices, ensure they are paired and connected. If using a physical jack, ensure it is fully inserted. You may need to update or reinstall the audio drivers via the Driver Update Utility.

7. Q: How do I connect to a new printer? A: All standard network printers are automatically pushed to devices when connected to the office network. For specialized printers, navigate to Settings > Devices > Printers & Scanners and click 'Add a printer or scanner.' Search for the printer by its asset tag (e.g., PRN-405-A).

8. Q: My docking station is failing to connect all peripherals (monitors, network, etc.). A: Disconnect the main USB-C/Thunderbolt cable connecting the laptop to the dock. Power cycle the docking station by unplugging its power cord for 10 seconds, then plug it back in. Reconnect the laptop cable. If the issue persists, the dock may need replacement; submit a ticket under 'Hardware Failure.'

## Network and Connectivity
### Wired and Wireless Access
9. Q: I cannot connect to the office Wi-Fi network. A: Ensure your Wi-Fi adapter is enabled. Verify you are selecting the correct network SSID (e.g., "Internal-Secure") and entering your current Active Directory password. If you are sure the password is correct, forget the network and attempt to reconnect.

10. Q: The wired (Ethernet) connection is showing as 'Unidentified Network'. A: Check that the Ethernet cable is securely plugged into both the wall jack and your computer/dock. Try swapping the cable with a known working one. If the issue remains, the wall port may be inactive; move to an adjacent port or submit a ticket to activate the current port, referencing the room and jack ID.

11. Q: My internet connection is intermittent or very slow. A: This often indicates high network traffic or a local configuration issue. Reboot your device first. If you are on Wi-Fi, try connecting via a wired connection to determine if the issue is specific to the wireless access point. If it affects multiple users in your area, submit a Severity 2 ticket under 'Infrastructure Outage.'

12. Q: How do I access the Guest Wi-Fi for a visitor? A: The Guest Wi-Fi SSID is "Visitor-Access". The password is reset weekly and is available on the internal IT Bulletin Board on the intranet. Access is capped at 4 hours per device, after which the guest must reauthenticate.

### VPN and Remote Access
13. Q: The Virtual Private Network (VPN) client is failing to connect. A: Check your internet connection first (local Wi-Fi or home network). Ensure the VPN client is the latest version (v5.1.4 or higher). Try logging in with your full email address as the username. If the connection times out, the VPN server may be undergoing maintenance; check the System Status Page.

14. Q: After connecting to the VPN, I cannot access internal file shares. A: Once connected to the VPN, ensure you are referencing the file share using the Fully Qualified Domain Name (FQDN), not the local name (e.g., use \\fileshare.domain.local\project instead of \\fileshare). If the FQDN fails, reboot and try again.

15. Q: I need access to a new server or network resource via VPN. A: Access to new resources requires manager approval. Submit a ticket under 'Access Request' detailing the server name or IP address and attach the manager's written approval (e.g., an email screenshot).

## Account and Security
### Password Management
16. Q: I have forgotten my password. How can I reset it? A: Use the Self-Service Password Reset (SSPR) portal available at [SSPR_Portal_Link]. You will need to answer the security questions you set up during onboarding. If you fail the security questions, you must contact the Help Desk directly during business hours.

17. Q: How often must I change my Active Directory password? A: Passwords must be changed every 90 calendar days. You will receive an email notification starting 14 days before expiration. The new password cannot be one of your last five used passwords and must meet the complexity requirements (12+ characters, including upper, lower, number, and symbol).

18. Q: My account is locked out after multiple failed login attempts. A: The system automatically locks accounts for 30 minutes after 5 failed attempts. Wait the required time and try again. If urgent, you can call the Help Desk to request an immediate manual unlock after verifying your identity.

19. Q: I need to set up Multi-Factor Authentication (MFA) on a new device. A: Log in to the Security Profile Portal. Select 'MFA Management' and click 'Add New Device.' You will be prompted to scan a QR code with the Authenticator App on your new mobile device. This must be done within 5 minutes of generating the code.

### Email and Access
20. Q: My Outlook email application keeps crashing or freezing. A: Try running Outlook in Safe Mode (Hold Ctrl while launching). If it runs fine in Safe Mode, the issue is likely a faulty add-in; disable recent or unnecessary add-ins. If the issue persists, run the Office Repair Tool from the Windows Control Panel.

21. Q: I need to request a shared mailbox or distribution list. A: Shared mailbox or DL requests must be submitted via the 'Email Configuration' ticket form. Include the required name, purpose, and list of initial members. Standard setup time is 2 business days.

22. Q: I am receiving an unusually high volume of spam or phishing attempts. A: Do not click any links. Use the 'Report Phish' button available in the Outlook toolbar for suspicious emails. If you believe your credentials may be compromised, change your password immediately and contact IT Security.

23. Q: I cannot send emails, and they are stuck in my Outbox. A: Check your network connectivity. If you are connected to the network, ensure your mailbox is not over its quota limit (default is 50GB). Delete large attachments or archive older emails to free up space.

## Software and Applications
### Installation and Licensing
24. Q: How do I install approved departmental software? A: All standard software (e.g., Adobe Reader, specialized accounting tools) is available in the Software Center (Windows) or Self-Service Portal (Mac). Locate the application and click 'Install.' Elevated privileges are not required.

25. Q: I need a new software application not listed in the Software Center. A: Unlisted software requires procurement and security vetting. Submit a ticket under 'Software Procurement Request.' Include the software name, vendor, cost, and a detailed business case justifying the need. Do not download or install unapproved software from the internet.

26. Q: My software license (e.g., graphic design suite) has expired. A: License renewals are managed centrally by IT. If your license has expired, the renewal process may be pending. Submit a ticket with the exact software name and the expiration date you see. IT will check the vendor license portal.

### Troubleshooting and Updates
27. Q: An application is consistently crashing and showing an error code. A: Note the exact error code or message (e.g., Error 0x80070002). Search the internal knowledge base first. If no results are found, submit a ticket with the error code and the exact steps taken leading up to the crash.

28. Q: How do I ensure my applications are up to date? A: Mandatory security updates are pushed automatically weekly on Tuesday nights. For non-critical updates, check the Help or About menu within the application and select 'Check for Updates.' Do not force manual updates unless instructed by IT.

29. Q: I need to clear the cache for a specific web-based application. A: Do not clear your entire browser cache, as this may impact other sites. Instead, in Chrome, go to Settings > Privacy and Security > Site Settings. Find the specific application URL and delete only the stored data and cookies for that site.

## Service and Support Procedures
30. Q: What is the priority level definition for support tickets? A:

- P1 (Critical): Loss of service impacting an entire team or critical business function (e.g., Payroll system down). Target Resolution: 4 hours.

- P2 (High): Loss of service impacting a single user's ability to work (e.g., cannot log in). Target Resolution: 8 business hours.

- P3 (Medium): Intermittent issues or non-critical application failures. Target Resolution: 24 business hours.

- P4 (Low): General requests, information, or training (e.g., monitor adjustment). Target Resolution: 48 business hours.

31. Q: How do I follow up on an existing ticket? A: Reply directly to the last email notification received from the ticketing system. Do not create a new ticket, as this will delay the resolution process.

32. Q: Can I bring my personal laptop to work for projects? A: No. Personal devices (BYOD) are strictly prohibited from connecting to the secure internal network or accessing confidential data, except for the use of the required MFA application on mobile phones.

33. Q: What is the procedure for offboarding an employee's access? A: Managers must submit an 'Offboarding Request' ticket at least 2 business days prior to the departure date. The request must specify the exact time of account deactivation and the mailbox forwarding address. Hardware must be collected by the manager and returned to IT within 24 hours.

## Cloud and File Storage
34. Q: Where should I store confidential company documents? A: All confidential or proprietary documents must be stored on the secure Cloud Drive platform or approved network shares. Never store confidential data on local device storage (the C: drive) or unapproved third-party cloud services.

35. Q: How do I restore a deleted file from the shared cloud drive? A: Navigate to the Cloud Drive web interface, select the folder where the file was located, and click 'Recycle Bin' or 'Trash.' Files are retained in the Recycle Bin for 90 days. Select the file and click 'Restore.'

36. Q: The shared drive is asking me for credentials constantly. A: This often indicates a cached credential issue. Open the Credential Manager (search in Windows Start Menu), find the entries related to the shared drive or domain, and delete them. Reboot your machine to force a fresh authentication.