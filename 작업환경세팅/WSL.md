#===================================================================================
#1 윈도 업데이트
#===================================================================================

https://www.microsoft.com/ko-kr/software-download/windows10 

> "지금업데이트" 클릭해서 업데이트파일 다운로드
> 무선랜 기준 약 1시간정도 소요되었음, 중간에 리붓 4~5번할때마다 SSD비번 입력해야되서 자리비우면 안됨.

#===================================================================================
#2 WSL2 설치
#===================================================================================

https://docs.microsoft.com/ko-kr/windows/wsl/install-win10
위에서 부터 세번째 명령어까지 POWERSHELL에서 실행하면 WSL2 설치완료

> 중간중간에 리붓 많이함


#===================================================================================
#3 UBUNTU18.04설치
#===================================================================================

MS스토어에서 설치하면되는데, ADJOIN 했으면 MS스토어 사용불가
아래 링크 참조
https://docs.microsoft.com/ko-kr/windows/wsl/install-manual



#===================================================================================
#4 윈도우터미널 설치
#===================================================================================
MS스토어에서 설치하면되는데, ADJOIN 했으면 MS스토어 사용불가
본인은 초코에서 설치하였음

초코: APT, YUM 같은 윈도우 패키지 매니저

https://chocolatey.org/install
choco install microsoft-windows-terminal


#===================================================================================
#5 WSL 메모리제한
#===================================================================================
vi /mnt/c/Users/LGCNS
-----------------------------------------------------
[wsl2]
memory=6GB
swap=0
localhostForwarding=true
-----------------------------------------------------


#===================================================================================
#6 DOCKER 설치
#===================================================================================

# DOCKER DESKTOP:
	데스크탑 방식은 알아서 하시되, 자사 인증서를 데스크탑상에서 설치해줘야함, 안해봄
	
# APT
	
	# 버전확인
	apt-cache madison package docker.io 
	
			docker.io | 19.03.6-0ubuntu1~18.04.1 | http://archive.ubuntu.com/ubuntu bionic-updates/universe amd64 Packages
		>>	docker.io | 18.09.7-0ubuntu1~18.04.4 | http://security.ubuntu.com/ubuntu bionic-security/universe amd64 Packages
			docker.io | 17.12.1-0ubuntu1 | http://archive.ubuntu.com/ubuntu bionic/universe amd64 Packages

	sudo apt-get install docker.io=18.09.7-0ubuntu1~18.04.4 -V
	
	# 인증서 설치:  https://wire.lgcns.com/confluence/pages/viewpage.action?pageId=35802456
	
		1. 회사 인증서를 윈도우 C 드라이브로 복사

			- 회사 인증서 다운로드 경로 : U-Cloud 접속 → S/W 다운로드 센터 → LG CNS SSL 인증서 → LG_CNS-CA.cer (1.15 KB)  다운로드

		2. 리눅스 /usr/share/ca-certificates/ 폴더로 이동하여 아무 폴더나 생성(ex: sudo mkdir extra)
		3. /usr/share/ca-certificates/extra 폴더로 윈도우 c 드라이브의 인증서를 복사
		   - sudo cp /c/LG_CNS-CA.cer /usr/share/ca-certificates/extra/LG_CNS-CA.cer
		4. /usr/share/ca-certificates/extra 폴더로 이동
		5. cer 인증서 확장자를 crt 로 변경
		   - sudo openssl x509 -inform PEM -in LG_CNS-CA.cer -out LG_CNS-CA.crt
		6. sudo dpkg-reconfigure ca-certificates 명령을 통해 리눅스에 CNS 인증서를 추가함
		   - 엔터 -> space 눌러서 LG_CNS-CA.crt 마킹 -> OK 
		   
		=> 끝 !
		
		
	

#===================================================================================
#7 WINDOWS TERMINAL 탬플릿 (주석조금 달았음)
#===================================================================================

// This file was initially generated by Windows Terminal 1.0.1401.0
// It should still be usable in newer versions, but newer versions might have additional
// settings, help text, or changes that you will not see unless you clear this file
// and let us generate a new one for you.

// To view the default settings, hold "alt" while clicking on the "Settings" button.
// For documentation on these settings, see: https://aka.ms/terminal-documentation
{
    "$schema": "https://aka.ms/terminal-profiles-schema",

    "defaultProfile": "{c6eaf9f4-32a7-5fdc-b5cf-066e8a4b1e40}",   // BASH의 UID 세팅

    // You can add more global application settings here.
    // To learn more about global settings, visit https://aka.ms/terminal-global-settings


    "copyOnSelect": true,										// MOBAXTERM 식 클릭복사
    "wordDelimiters": "()':,;< >~!@#$%^&*|+=[]{}~?/",			// 더블클릭시 선택영역 구분자
    
	// If enabled, formatted data is also copied to your clipboard
    "copyFormatting": false,

    // A profile specifies a command to execute paired with information about how it should look and feel.
    // Each one of them will appear in the 'New Tab' dropdown,
    //   and can be invoked from the commandline with 'wt.exe -p xxx'
    // To learn more about profiles, visit https://aka.ms/terminal-profile-settings
    "profiles":
    {
        "defaults":
        {
            // Put settings here that you want to apply to all profiles.
        },
        "list":
        [
            {
                // Make changes here to the powershell.exe profile.
                "guid": "{61c54bbd-c2c6-5271-96e7-009a87ff44bf}",
                "name": "Windows PowerShell",
                "commandline": "powershell.exe",
                "hidden": false
            },
            {
                // Make changes here to the cmd.exe profile.
                "guid": "{0caa0dad-35be-5f56-a8ff-afceeeaa6101}",
                "name": "명령 프롬프트",
                "commandline": "cmd.exe",
                "hidden": false
            },
            {
                "guid": "{c6eaf9f4-32a7-5fdc-b5cf-066e8a4b1e40}",
                "hidden": false,
                "name": "Ubuntu-18.04",
                "source": "Windows.Terminal.Wsl",
			  // CUSTOM====================================
				"colorScheme": "One Half Dark",
				"useAcrylic": true,	// The transparency only applies to focused windows due to OS limitations.
				"acrylicOpacity": 0.8,
				"fontSize": 10
			  //======================================================

            },
            {
                "guid": "{b453ae62-4e3d-5e58-b989-0a998ec441b8}",
                "hidden": false,
                "name": "Azure Cloud Shell",
                "source": "Windows.Terminal.Azure"
            }
        ]
    },

    // Add custom color schemes to this array.
    // To learn more about color schemes, visit https://aka.ms/terminal-color-schemes
    "schemes": [],


    // Add custom keybindings to this array.
    // To unbind a key combination from your defaults.json, set the command to "unbound".
    // To learn more about keybindings, visit https://aka.ms/terminal-keybindings
    "keybindings":
    [
        // Copy and paste are bound to Ctrl+Shift+C and Ctrl+Shift+V in your defaults.json.
        // These two lines additionally bind them to Ctrl+C and Ctrl+V.
        // To learn more about selection, visit https://aka.ms/terminal-selection
        { "command": {"action": "copy", "singleLine": false }, "keys": "ctrl+c" },
        { "command": "paste", "keys": "ctrl+v" },

        // Press Ctrl+Shift+F to open the search box
        { "command": "find", "keys": "ctrl+shift+f" },

        // Press Alt+Shift+D to open a new pane.
        // - "split": "auto" makes this pane open in the direction that provides the most surface area.
        // - "splitMode": "duplicate" makes the new pane use the focused pane's profile.
        // To learn more about panes, visit https://aka.ms/terminal-panes
        { "command": { "action": "splitPane", "split": "auto", "splitMode": "duplicate" }, "keys": "alt+shift+d" }
    ]
}



