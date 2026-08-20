package main

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/spf13/cobra"
	"github.com/vibe-coding-labs/JoyCode2Api/pkg/auth"
	"github.com/vibe-coding-labs/JoyCode2Api/pkg/joycode"
)

var configCmd = &cobra.Command{
	Use:     "config",
	Short:   "显示当前配置信息",
	Long:    "显示已解析的凭据来源、默认设置和服务安装状态。",
	GroupID: "query",
	Example: `  joycode-proxy config

  # 查看指定凭据的配置
  joycode-proxy -k <ptkey> -u <userid> config`,
	RunE: func(cmd *cobra.Command, args []string) error {
		fmt.Println("JoyCode Proxy Configuration")
		fmt.Println("============================")

		// Credentials
		fmt.Println()
		fmt.Println("  Credentials:")
		if ptKey != "" && userID != "" {
			fmt.Printf("    Source:    flags\n")
			fmt.Printf("    UserID:    %s\n", userID)
			fmt.Printf("    PtKey:     %s...%s\n", ptKey[:8], ptKey[len(ptKey)-4:])
		} else {
			creds, err := auth.LoadFromSystem()
			if err != nil {
				fmt.Printf("    Source:    not available (%s)\n", err)
			} else {
				fmt.Printf("    Source:    auto-detected\n")
				fmt.Printf("    UserID:    %s\n", creds.UserID)
				fmt.Printf("    PtKey:     %s...%s\n", creds.PtKey[:8], creds.PtKey[len(creds.PtKey)-4:])
			}
		}

		// API
		fmt.Println()
		fmt.Println("  API:")
		fmt.Printf("    Base URL:       %s\n", joycode.BaseURL)
		fmt.Printf("    Default Model:  %s\n", joycode.DefaultModel)

		// Server
		fmt.Println()
		fmt.Println("  Server:")
		fmt.Printf("    Default Host:  %s\n", "127.0.0.1")
		fmt.Printf("    Default Port:  %d\n", 34891)

		// Service
		fmt.Println()
		fmt.Println("  Service:")
		home, _ := os.UserHomeDir()
		plistPath := filepath.Join(home, "Library", "LaunchAgents", plistName)
		if _, err := os.Stat(plistPath); os.IsNotExist(err) {
			fmt.Println("    Installed: no")
		} else {
			fmt.Println("    Installed: yes")
			fmt.Printf("    Plist:     %s\n", plistPath)
		}

		// Models. No fetcher is registered outside `serve`, so this prints the
		// built-in snapshot; use `joycode-proxy models` for the live list.
		fmt.Println()
		fmt.Println("  Available Models (built-in snapshot):")
		for _, m := range joycode.DefaultRegistry.Visible() {
			suffix := ""
			if m.APIName() == joycode.DefaultModel {
				suffix = " (default)"
			}
			fmt.Printf("    - %s%s\n", m.DisplayName(), suffix)
		}

		return nil
	},
}

func init() {
	rootCmd.AddCommand(configCmd)
}
